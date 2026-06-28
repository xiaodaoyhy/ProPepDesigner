import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import pandas as pd
from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig
from pepinvent.supervised_learning.trainer.base_trainer import BaseTrainer
from pepinvent.supervised_learning.trainer.create_vocabulary import VocabularyMaker
from pepinvent.supervised_learning.utils.chem import tanimoto_similarity
from pepinvent.supervised_learning.utils.file import make_directory
from pepinvent.supervised_learning.utils.log import progress_bar
from pepinvent.supervised_learning.utils.log import get_logger
from reinvent_models.mol2mol.models.encode_decode.model import EncoderDecoder
from reinvent_models.mol2mol.models.module.decode import decode
from reinvent_models.mol2mol.models.module.label_smoothing import LabelSmoothing
from reinvent_models.mol2mol.models.module.noam_opt import NoamOpt
from reinvent_models.mol2mol.models.vocabulary import SMILESTokenizer
from reinvent_models.mol2mol.mol2mol_model import Mol2MolModel
from pepinvent.reinforcement.chemistry import Chemistry

class TransformerTrainer(BaseTrainer):
    def __init__(self, opt: ArchitectureConfig):
        super().__init__(opt)

    class _EncoderDecoderWithGenerator(torch.nn.Module):
        """
        Wrap EncoderDecoder so that `forward()` includes the generator.

        This is important for DDP: if generator parameters are used outside the DDP-wrapped module's forward,
        DDP can throw "Expected to mark a variable ready only once" (especially with unused-parameter detection).
        """
        def __init__(self, network: EncoderDecoder):
            super().__init__()
            self.network = network
            self.generator = network.generator

        def forward(self, src, tgt, src_mask, tgt_mask, temperature: float = 1.0):
            out = self.network.forward(src, tgt, src_mask, tgt_mask)
            return self.generator(out, temperature=temperature)

        def encode(self, src, src_mask):
            return self.network.encode(src, src_mask)

        def decode(self, memory, src_mask, tgt, tgt_mask):
            return self.network.decode(memory, src_mask, tgt, tgt_mask)

    def _ddp_param_sanity_check(self, model, device, step: int):
        """
        Periodically compare a small slice of a parameter across ranks.
        If DDP is working correctly, parameters should stay in sync (diff ~ 0).
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        if not getattr(self._config, "ddp_param_check_enabled", False):
            return

        interval = max(int(getattr(self._config, "ddp_param_check_interval_steps", 2000)), 1)
        if step % interval != 0:
            return

        numel = max(int(getattr(self._config, "ddp_param_check_numel", 2048)), 1)
        rtol = float(getattr(self._config, "ddp_param_check_rtol", 1e-5))
        atol = float(getattr(self._config, "ddp_param_check_atol", 1e-7))

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Pick the first trainable parameter as a representative.
        param = None
        for p in model.parameters():
            if p.requires_grad:
                param = p
                break
        if param is None:
            return

        flat = param.detach().reshape(-1)
        k = min(numel, flat.numel())
        sample = flat[:k].to(device=device, dtype=torch.float32).contiguous()

        gathered = [torch.empty_like(sample) for _ in range(world_size)]
        dist.all_gather(gathered, sample)

        if rank == 0:
            ref = gathered[0]
            max_abs_diffs = []
            for r in range(1, world_size):
                d = torch.max(torch.abs(gathered[r] - ref)).item()
                max_abs_diffs.append(d)
            max_abs_diff = max(max_abs_diffs) if max_abs_diffs else 0.0
            ref_max = torch.max(torch.abs(ref)).item()
            tol = atol + rtol * ref_max
            if max_abs_diff > tol:
                self.LOG.warning(
                    f"[DDP_CHECK] step={step} param_slice_max_abs_diff={max_abs_diff:.3e} "
                    f"(tol={tol:.3e}, rtol={rtol:.1e}, atol={atol:.1e}, ref_max={ref_max:.3e})"
                )
            else:
                self.LOG.info(
                    f"[DDP_CHECK] step={step} param_slice_max_abs_diff={max_abs_diff:.3e} "
                    f"(tol={tol:.3e})"
                )

    def get_model(self, vocab) -> Mol2MolModel:
        # Fresh training / fine-tuning start
        if self._config.starting_epoch == 1:
            # Fine-tune from external pretrained model if provided
            if getattr(self._config, "pretrained_model_path", None):
                model = Mol2MolModel.load_from_file(self._config.pretrained_model_path)
                return model

            if vocab is None:
                raise ValueError("vocab must be provided when starting_epoch == 1")
            vocab_size = len(vocab.tokens())
            network = EncoderDecoder(vocab_size, num_layers=self._config.N, model_dimension=self._config.d_model,
                                   feedforward_dimension=self._config.d_ff, num_heads=self._config.H,
                                   dropout=self._config.dropout)
            model = Mol2MolModel(vocabulary=vocab, network=network,
                                 max_sequence_length=self._config.max_sequence_length,
                                 no_cuda=not self._config.use_cuda)
        else:
            file_name = os.path.join(self.save_path, f'checkpoint/model_{self._config.starting_epoch - 1}.ckpt')
            model = Mol2MolModel.load_from_file(file_name)

        return model

    def _initialize_optimizer(self, model):
        optim = NoamOpt(model.src_embed[0].d_model, self._config.factor, self._config.warmup_steps,
                        torch.optim.Adam(model.parameters(), lr=0,
                                         betas=(self._config.adam_beta1, self._config.adam_beta2),
                                         eps=self._config.adam_eps))
        return optim

    def _load_optimizer_from_epoch(self, model, file_name, device):
        # load optimization
        checkpoint = torch.load(file_name, map_location=device)
        optim_dict = checkpoint['optimizer_state_dict']
        optim = NoamOpt(optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'],
                        torch.optim.Adam(model.parameters(), lr=0))
        optim.load_state_dict(optim_dict)

        return optim

    def _load_optimizer_from_path(self, model, file_name, device):
        # Same format as our saved optimizer checkpoints
        checkpoint = torch.load(file_name, map_location=device)
        optim_dict = checkpoint['optimizer_state_dict']
        optim = NoamOpt(
            optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'],
            torch.optim.Adam(model.parameters(), lr=0)
        )
        optim.load_state_dict(optim_dict)
        return optim

    def _extract_new_tokens_from_csv(self, csv_paths, tokenizer: SMILESTokenizer, existing_vocab) -> list:
        """
        Scan CSV(s) and return a sorted list of tokens not present in existing_vocab.
        Deterministic ordering is important so all DDP ranks extend vocab identically.
        """
        tokens = set()
        for path in csv_paths:
            df_iter = pd.read_csv(path, iterator=True, chunksize=200)
            for frame in df_iter:
                for col in ("Source_Mol", "Target_Mol"):
                    if col not in frame.columns:
                        continue
                    for smi in frame[col].astype(str).tolist():
                        toks = tokenizer.tokenize(smi, with_begin_and_end=False)
                        tokens.update(toks)

        existing = set(existing_vocab.tokens())
        new_tokens = sorted([t for t in tokens if t not in existing])
        return new_tokens

    def _resize_vocab_dependent_layers(self, network: EncoderDecoder, new_vocab_size: int):
        """
        Resize src/tgt embeddings and generator projection to match new vocabulary size.
        Copies old weights; initializes new rows/outputs.
        """
        old_vocab_size = network.vocabulary_size
        if new_vocab_size == old_vocab_size:
            return
        if new_vocab_size < old_vocab_size:
            raise ValueError(f"new_vocab_size ({new_vocab_size}) < old_vocab_size ({old_vocab_size}) is not supported.")

        device = next(network.parameters()).device
        d_model = network.model_dimension

        # --- Embeddings ---
        for embed in (network.src_embed[0].lut, network.tgt_embed[0].lut):
            old_weight = embed.weight.data
            new_embed = torch.nn.Embedding(new_vocab_size, d_model).to(device)
            torch.nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
            new_embed.weight.data[:old_vocab_size].copy_(old_weight)
            embed.weight = torch.nn.Parameter(new_embed.weight.data)
            embed.num_embeddings = new_vocab_size

        # --- Generator ---
        old_proj = network.generator.proj
        new_proj = torch.nn.Linear(d_model, new_vocab_size).to(device)
        torch.nn.init.xavier_uniform_(new_proj.weight)
        torch.nn.init.zeros_(new_proj.bias)
        new_proj.weight.data[:old_vocab_size].copy_(old_proj.weight.data)
        new_proj.bias.data[:old_vocab_size].copy_(old_proj.bias.data)
        network.generator.proj = new_proj

        network.vocabulary_size = new_vocab_size

    def get_optimization(self, model, device):
        if self._config.starting_epoch == 1:
            # Fine-tuning warm start: optionally load optimizer, or reset.
            if getattr(self._config, "pretrained_model_path", None):
                if getattr(self._config, "reset_optimizer", False):
                    optim = self._initialize_optimizer(model)
                else:
                    opt_path = getattr(self._config, "pretrained_optimizer_path", None)
                    if opt_path:
                        optim = self._load_optimizer_from_path(model, opt_path, device)
                    else:
                        optim = self._initialize_optimizer(model)
            else:
                optim = self._initialize_optimizer(model)
        else:
            # load optimization
            file_name = os.path.join(self.save_path, f'checkpoint/optimizer_{self._config.starting_epoch - 1}.ckpt')
            optim = self._load_optimizer_from_epoch(model, file_name, device)
        return optim

    def execute(self):
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        is_main = rank == 0
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        # 设置默认设备
        torch.cuda.set_device(device)

        # Avoid multiple ranks writing to the same log/tensorboard files.
        # - per-rank log file
        self.LOG = get_logger(
            name=f"train_model_rank{rank}",
            log_path=os.path.join(self.save_path, f"train_model_rank{rank}.log")
        )
        # - only rank0 writes tensorboard (others no-op)
        if not is_main:
            class _NoOpWriter:
                def add_scalar(self, *args, **kwargs):  # noqa: ANN001
                    return None
                def add_scalars(self, *args, **kwargs):  # noqa: ANN001
                    return None
                def flush(self):  # noqa: D401
                    return None
                def close(self):
                    return None
            self.summary_writer = _NoOpWriter()

        # IMPORTANT vocab rules:
        # - Fresh training: build vocabulary from current data.
        # - Resume (starting_epoch != 1): MUST use vocabulary stored inside the checkpointed model.
        # - Fine-tuning from pretrained_model_path: default is to use pretrained vocab; optionally extend it
        #   and resize embeddings/generator safely.
        vocab = None
        tokenizer = SMILESTokenizer()
        if self._config.starting_epoch == 1 and not getattr(self._config, "pretrained_model_path", None):
            vocabulary_maker = VocabularyMaker()
            vocab = vocabulary_maker.create_vocabulary(self._config.training_data_path, self._config.validation_data_path)

        model = self.get_model(vocab)
        vocab = model.vocabulary

        # Optional: extend vocabulary for fine-tuning with new tokens
        if self._config.starting_epoch == 1 and getattr(self._config, "pretrained_model_path", None) and getattr(self._config, "extend_vocabulary", False):
            # Compute new tokens on rank0 and broadcast, to guarantee all ranks update identically.
            new_tokens = None
            if is_main:
                new_tokens = self._extract_new_tokens_from_csv(
                    [self._config.training_data_path, self._config.validation_data_path],
                    tokenizer=tokenizer,
                    existing_vocab=vocab
                )
                self.LOG.info(f"Fine-tune: extending vocabulary with {len(new_tokens)} new tokens.")
            obj_list = [new_tokens]
            dist.broadcast_object_list(obj_list, src=0)
            new_tokens = obj_list[0] or []
            if len(new_tokens) > 0:
                vocab.update(new_tokens)
                self._resize_vocab_dependent_layers(model.network, new_vocab_size=len(vocab))

        vocab_size = len(vocab.tokens())
        dataloader_train = self.initialize_dataloader(
            self._config.training_data_path, self._config.batch_size, vocab, sampler=DistributedSampler
        )
        # validation_batch_size = self._config.batch_size // 2  # 使用训练batch size的一半,节省显存
        validation_batch_size = self._config.batch_size  # 使用训练batch size的一半,节省显存

        dataloader_validation = self.initialize_dataloader(
            self._config.validation_data_path, validation_batch_size, vocab, sampler=DistributedSampler
        )

        model.network = model.network.to(device)
        optimization = self.get_optimization(model.network, device)

        # IMPORTANT:
        # Wrap network so generator is executed inside forward() (DDP-safe).
        train_module = self._EncoderDecoderWithGenerator(model.network).to(device)

        # Keep DDP wrapper for forward/backward; do NOT replace it with `.module`.
        ddp_network = torch.nn.parallel.DistributedDataParallel(
            train_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )

        pad_idx = self._config.padding_value
        criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=self._config.label_smoothing)
        criterion = criterion.to(device)  # 确保损失函数也在正确设备上
        
        
        # 内存优化设置
        torch.backends.cudnn.benchmark = True  # 优化cudnn性能
        torch.cuda.empty_cache()  # 清理初始GPU缓存

        try:
            # Train epoch
            for epoch in range(self._config.starting_epoch, self._config.starting_epoch + self._config.num_epoch):
                # Ensure DistributedSampler shuffles differently each epoch
                if hasattr(dataloader_train, "sampler") and hasattr(dataloader_train.sampler, "set_epoch"):
                    dataloader_train.sampler.set_epoch(epoch)
                if hasattr(dataloader_validation, "sampler") and hasattr(dataloader_validation.sampler, "set_epoch"):
                    dataloader_validation.sampler.set_epoch(epoch)

                self.LOG.info("EPOCH #%d Training start", epoch)
                ddp_network.train()
                loss_epoch_train = self.train_epoch(epoch, dataloader_train, ddp_network, criterion, optimization, device)
                self.LOG.info("EPOCH #%d Training end", epoch)

                if is_main and epoch >= 8:
                    self.save(model, optimization, epoch, vocab_size)

                self.LOG.info("EPOCH #%d Validation start", epoch)
                ddp_network.eval()
                loss_epoch_validation, accuracy, token_accuracy, sim_avg, nll_mean = self.validation_stat(
                    dataloader_validation, ddp_network, criterion, device, vocab
                )
                self.LOG.info("EPOCH #%d Validation end", epoch)

                if is_main:
                    self.LOG.info(
                        "EPOCH #{}, Train loss, Validation loss, identity_accuracy, token_accuracy, sim_avg: {}, {}, {}, {}, {}".format(
                            epoch, round(loss_epoch_train, 5), round(loss_epoch_validation, 5),
                            round(accuracy, 5), round(token_accuracy, 5), round(sim_avg, 5)))
                    self.LOG.info("EPOCH #{}, Mean NLL: {}".format(epoch, nll_mean))

                    # 获取当前学习率
                    current_lr = optimization._rate
                    self.LOG.info("EPOCH #{} Current Learning Rate: {:.2e}".format(epoch, current_lr))
                    self.to_tensorboard(
                        loss_epoch_train, loss_epoch_validation, accuracy, token_accuracy, sim_avg, epoch,
                        nll_mean, learning_rate=current_lr
                    )
        finally:
            try:
                self.summary_writer.close()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                pass

    def train_epoch(self, epoch, dataloader, model, criterion, optimization, device):
        pad = self._config.padding_value
        total_loss, total_tokens = 0.0, 0.0
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        show_progress = rank == 0

        # 记录开始训练时的学习率
        start_lr = optimization._rate
        self.LOG.info(f"Epoch #{epoch} start LR: {start_lr:.2e}")

        for i, batch in enumerate(progress_bar(dataloader, total=len(dataloader), disable=not show_progress)):
            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            # Compute loss
            log_probs = model.forward(src, trg, src_mask, trg_mask)
            loss = criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)),
                trg_y.contiguous().view(-1)
            ) / ntokens

            loss.backward()
            optimization.step()
            optimization.optimizer.zero_grad()

            total_tokens += ntokens
            total_loss += float(loss.detach().cpu().item()) * ntokens

            # DDP parameter sync sanity check (lightweight, configurable)
            try:
                self._ddp_param_sanity_check(model, device, step=int(optimization._step))
            except Exception as e:
                # Don't crash training because of a debug check.
                self.LOG.warning(f"[DDP_CHECK] failed at step={getattr(optimization, '_step', 'NA')}: {e}")


            # 记录学习率
            # 每个batch后记录学习率变化
            if i % 1500 == 0:  # 每500个batch记录一次
                current_lr = optimization._rate
                current_step = optimization._step
                self.LOG.info(f"\nEpoch #{epoch}: Step {current_step}, LR: {current_lr:.2e}")
                torch.cuda.empty_cache()
        
        # 记录epoch结束时的学习率
        end_lr = optimization._rate
        self.LOG.info(f"\nEpoch #{epoch} end LR: {end_lr:.2e}")
        # Reduce across ranks to get global train loss
        if dist.is_available() and dist.is_initialized():
            totals = torch.tensor([total_loss, total_tokens], device=device, dtype=torch.float64)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss, total_tokens = float(totals[0].item()), float(totals[1].item())
        loss_epoch = total_loss / max(total_tokens, 1.0)
        return loss_epoch

    def _get_model_parameters(self, vocab_size):
        return {
            'vocab_size': vocab_size,
            'N': self._config.N,
            'd_model': self._config.d_model,
            'd_ff': self._config.d_ff,
            'H': self._config.H,
            'dropout': self._config.dropout
        }

    def save(self, model: Mol2MolModel, optim, epoch, vocab_size):
        """
        Saves the model, optimizer and model hyperparameters
        """
        save_dict = {
            'optimizer_state_dict': optim.save_state_dict(),
        }

        file_name = os.path.join(self.save_path, f'checkpoint/optimizer_{epoch}.ckpt')
        make_directory(file_name, is_dir=False)
        torch.save(save_dict, file_name)

        file_name = os.path.join(self.save_path, f'checkpoint/model_{epoch}.ckpt')
        model.save_to_file(file_name)

    def validation_stat(self, dataloader, model, criterion, device, vocab):
        pad = self._config.padding_value
        total_loss, total_n_trg, total_tokens, n_correct, n_correct_token = 0, 0, 0, 0, 0
        sim_sum, sim_count = 0.0, 0.0
        nll_sum, nll_count = 0.0, 0.0
        tokenizer = SMILESTokenizer()
        chem = Chemistry()
        sim_pairs_done = 0
        sim_enabled = bool(getattr(self._config, "validation_similarity_enabled", True))
        sim_max_pairs = int(getattr(self._config, "validation_similarity_max_pairs_per_rank", 200))
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        show_progress = rank == 0

        # 清理显存
        torch.cuda.empty_cache()

        for i, batch in enumerate(progress_bar(dataloader, total=len(dataloader), disable=not show_progress)):
            # 每10个batch清理一次显存
            if i % 5 == 0:
                torch.cuda.empty_cache()

            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            with torch.no_grad():
                log_probs = model.forward(src, trg, src_mask, trg_mask)
                loss = criterion(
                    log_probs.contiguous().view(-1, log_probs.size(-1)),
                    trg_y.contiguous().view(-1)
                ) / ntokens
                total_loss += float(loss.detach().cpu().item()) * ntokens
                total_tokens += ntokens
                # Decode
                max_length_target = self._config.max_sequence_length
                # `decode()` expects the underlying model module to expose encode/decode/generator.
                # When wrapped in DDP, these methods live on `model.module`.
                decode_model = model.module if hasattr(model, "module") else model
                smiles, nlls = decode(decode_model, src, src_mask, max_length_target, device, decode_type='greedy')
                # aggregate NLL across samples (not batch-means)
                nll_sum += float(torch.sum(nlls).detach().cpu().item())
                nll_count += float(nlls.numel())
                del nlls  # 删除GPU张量

                # Compute accuracy_harsh, accuracy_smooth
                # 优化策略：核心数据移到CPU，保持高效处理
                batch_size = trg.size(0)
                
                # 批量移动到CPU
                smiles_cpu = smiles.cpu()
                trg_cpu = trg.cpu()
                src_cpu = src.cpu()
                
                for j in range(batch_size):
                    seq_tokens = smiles_cpu[j, :].numpy()
                    target_tokens = trg_cpu[j].numpy()

                    # 字符串解码
                    target = tokenizer.untokenize(vocab.decode(target_tokens))
                    seq = tokenizer.untokenize(vocab.decode(seq_tokens))
                    
                    # 准确率统计（简单比较，CPU很快）
                    if seq == target:
                        n_correct += 1

                    # token accuracy
                    for k in range(len(target)):
                        if k < len(seq) and seq[k] == target[k]:
                            n_correct_token += 1

                    # Similarity:
                    # - `seq` / `target` are peptide-representation strings (may contain '|' and represent only masked AAs)
                    # - Convert to full peptide SMILES by filling into the masked source, then compute RDKit Tanimoto.
                    start_ind = 0
                    source_seq = tokenizer.untokenize(vocab.decode(src[j].cpu().numpy()[start_ind:]))
                    full_pred = chem.fill_source_peptide(source_seq, seq)
                    full_true = chem.fill_source_peptide(source_seq, target)
                    if sim_enabled and sim_pairs_done < sim_max_pairs:
                        if full_pred != 'none' and full_true != 'none':
                            sim = tanimoto_similarity(full_pred, full_true)
                            if sim is not None:
                                sim_sum += float(sim)
                                sim_count += 1.0
                                sim_pairs_done += 1
                    # Debug printing can severely slow down validation and scramble tqdm output under DDP.
                    # Only print on rank0, and keep it very sparse.
                    if rank == 0 and j % 20 == 0 and i == 0:
                        print(f"{source_seq.count('|')},{source_seq.count('?')}, source_seq:{source_seq}\n")
                        print(f"seq_tokens.shape={seq_tokens.shape}")
                        print(f"seq:{seq}")
                        print(f"target_tokens.shape={target_tokens.shape}")
                        print(f"{target.count('|')},{target.count('?')}, target:{target}")

                # 高效内存清理
                del smiles_cpu, trg_cpu, src_cpu, smiles, log_probs

            # number of samples in current batch
            n_trg = trg.size()[0]
            # total samples
            total_n_trg += n_trg

        # Reduce across ranks to get global validation stats
        if dist.is_available() and dist.is_initialized():
            totals = torch.tensor(
                [total_loss, total_tokens, total_n_trg, n_correct, n_correct_token, sim_sum, sim_count, nll_sum, nll_count],
                device=device,
                dtype=torch.float64
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss = float(totals[0].item())
            total_tokens = float(totals[1].item())
            total_n_trg = float(totals[2].item())
            n_correct = float(totals[3].item())
            n_correct_token = float(totals[4].item())
            sim_sum = float(totals[5].item())
            sim_count = float(totals[6].item())
            nll_sum = float(totals[7].item())
            nll_count = float(totals[8].item())

        # Accuracy
        accuracy_harsh = n_correct / max(total_n_trg, 1.0)
        loss_epoch = total_loss / max(total_tokens, 1.0)
        token_accuracy = n_correct_token / max(total_tokens, 1.0)
        sim_avg = sim_sum / sim_count if sim_count > 0 else 0.0
        nll_mean = nll_sum / max(nll_count, 1.0)
        return loss_epoch, accuracy_harsh, token_accuracy, sim_avg, nll_mean
