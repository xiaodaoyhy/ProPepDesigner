import numpy as np
from sklearn.model_selection import train_test_split
from Bio.Align import PairwiseAligner
from typing import Dict, List, Sequence, Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator



def random_split(
    n_samples: int,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split sample indices into train / val / test.

    Parameters
    ----------
    n_samples : int
        Total number of samples
    val_ratio : float
        Validation set fraction of all data
    test_ratio : float
        Test set fraction of all data
    seed : int
        Random seed
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1

    indices = np.arange(n_samples)

    # First split train vs. (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    # Then split (val + test) into val / test
    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )

    return train_idx, val_idx, test_idx






def label_stratify_split(
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    n_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple multi-label stratified split based on activity values:
    - For each task, bin activity by quantiles into n_bins (NaN ignored)
    - Combine per-sample bins across tasks into a "label pattern" for stratification

    Note: This is an approximate multi-label stratification, used only to keep
          train/val/test similar in activity distribution patterns.

    Parameters
    ----------
    y : np.ndarray, shape [n_samples, n_tasks], may contain NaN
    val_ratio, test_ratio, seed, n_bins : same as above
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1
    n_samples, n_tasks = y.shape

    # 1. Bin each task; NaN marked as -1
    bins_per_task = np.full_like(y, fill_value=-1, dtype=int)

    for j in range(n_tasks):
        col = y[:, j]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            # No labels at all for this task
            continue

        # Compute quantile boundaries
        qs = np.linspace(0, 100, n_bins + 1)
        try:
            quantiles = np.nanpercentile(col[mask], qs)
        except Exception:
            # Skip binning if quantile computation fails
            continue

        # Slightly perturb boundaries to avoid duplicates
        quantiles[0] -= 1e-6
        quantiles[-1] += 1e-6

        # Assign bin indices
        # np.digitize returns 1..n_bins; subtract 1 to get 0..n_bins-1
        bins = np.digitize(col[mask], quantiles[1:-1])  # use inner boundaries
        bins_per_task[mask, j] = bins

    # 2. Combine per-sample bins into string labels for stratification
    labels = np.array(
        ["|".join(map(str, row)) for row in bins_per_task],
        dtype=object,
    )

    # 3. Handle singleton "rare labels": put directly in train, skip stratification
    indices = np.arange(n_samples)
    _, counts = np.unique(labels, return_counts=True)
    label_to_count = {lab: cnt for lab, cnt in zip(np.unique(labels), counts)}
    is_rare = np.array([label_to_count[lab] < 2 for lab in labels])

    rare_idx = indices[is_rare]
    strat_indices = indices[~is_rare]
    strat_labels = labels[~is_rare]

    # 4. Stratified split on remaining samples: train / (val + test)
    if strat_indices.size > 0:
        unique_labels = np.unique(strat_labels)
        # Fall back to non-stratified split if too few classes or too few samples per class
        if unique_labels.size > 1:
            _, rem_counts = np.unique(strat_labels, return_counts=True)
            if rem_counts.min() >= 2:
                stratify_labels = strat_labels
            else:
                stratify_labels = None
        else:
            stratify_labels = None

        train_idx_rem, temp_idx = train_test_split(
            strat_indices,
            test_size=val_ratio + test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_labels,
        )
        # Final train set = stratified train + all rare samples
        train_idx = np.concatenate([rare_idx, train_idx_rem]) if rare_idx.size > 0 else train_idx_rem
    else:
        # All samples are rare labels; fall back to random split only
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_ratio + test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    if len(temp_idx) == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)

    # 5. Split temp (val + test) into val / test
    temp_labels = labels[temp_idx]
    unique_temp_labels, temp_counts = np.unique(temp_labels, return_counts=True)
    # Stratify only when num_classes > 1 and each class has >= 2 samples; else non-stratified
    if unique_temp_labels.size > 1 and temp_counts.min() >= 2:
        stratify_temp = temp_labels
    else:
        stratify_temp = None

    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )

    return train_idx, val_idx, test_idx





# ============================================================================
# ===== sequence-cluster split starts here ===============================
# ============================================================================

# 聚类划分数据集与seed无关，所以把聚类结果缓存，节省计算
_SEQUENCE_CLUSTER_CACHE = {}
_ECFP_CLUSTER_CACHE = {}

# 用于构建序列簇的最小并集查找结构。
class _UnionFind:
    """Minimal union-find structure used to build connected components."""
    
    def __init__(self, n_items: int):
        self.parent = np.arange(n_items, dtype=int)
        self.rank = np.zeros(n_items, dtype=int)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return int(item)

    def union(self, item_a: int, item_b: int) -> None:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def _encode_token_sequences(sequences: Sequence[Sequence[str]]) -> List[str]:
    """
    Encode each residue token as one Unicode character.

    This lets Bio.Align perform a fast global alignment while preserving ncAA
    identity: e.g. Aib != A, dE != E, and aMeL != L.
    """
    normalized_sequences = [[str(token) for token in seq] for seq in sequences]
    vocabulary = sorted({token for seq in normalized_sequences for token in seq})
    if len(vocabulary) > 6400:
        raise ValueError("Too many unique residue tokens for sequence encoding.")

    token_to_char = {
        token: chr(0xE000 + token_idx)
        for token_idx, token in enumerate(vocabulary)
    }
    return ["".join(token_to_char[token] for token in seq) for seq in normalized_sequences]


def _global_token_identity(
    encoded_seq_a: str,
    encoded_seq_b: str,
    aligner: PairwiseAligner,
) -> float:
    """
    Calculate token identity from a Needleman-Wunsch global alignment.

    Identity = number of exactly matched residue tokens / alignment length,
    where alignment length includes mismatch and gap columns.
    """
    if not encoded_seq_a and not encoded_seq_b:
        return 1.0
    if not encoded_seq_a or not encoded_seq_b:
        return 0.0

    alignment = aligner.align(encoded_seq_a, encoded_seq_b)[0]
    coordinates = alignment.coordinates
    matches = 0
    alignment_length = 0

    for segment_idx in range(coordinates.shape[1] - 1):
        a_start = int(coordinates[0, segment_idx])
        a_end = int(coordinates[0, segment_idx + 1])
        b_start = int(coordinates[1, segment_idx])
        b_end = int(coordinates[1, segment_idx + 1])
        a_step = a_end - a_start
        b_step = b_end - b_start

        if a_step > 0 and b_step > 0:
            # In an aligned segment both sequences advance by the same amount.
            segment_length = min(a_step, b_step)
            matches += sum(
                encoded_seq_a[a_start + offset] == encoded_seq_b[b_start + offset]
                for offset in range(segment_length)
            )
            alignment_length += max(a_step, b_step)
        else:
            # One sequence advances while the other contains a gap.
            alignment_length += a_step + b_step

    return matches / alignment_length if alignment_length > 0 else 0.0


def _build_sequence_clusters(
    sequences: Sequence[Sequence[str]],
    identity_threshold: float,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Build connected-component clusters from pairwise sequence identity."""
    if not 0 < identity_threshold <= 1:
        raise ValueError("identity_threshold must be in the interval (0, 1].")

    n_samples = len(sequences)
    if n_samples == 0:
        raise ValueError("Cannot split an empty sequence collection.")

    encoded_sequences = _encode_token_sequences(sequences)
    sequence_lengths = np.asarray([len(seq) for seq in encoded_sequences], dtype=int)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -1.0

    union_find = _UnionFind(n_samples)
    print(
        f"Building token-aware sequence clusters for {n_samples} samples "
        f"at identity >= {identity_threshold:.2f} ..."
    )

    for sample_i in range(n_samples - 1):
        len_i = sequence_lengths[sample_i]
        for sample_j in range(sample_i + 1, n_samples):
            len_j = sequence_lengths[sample_j]

            # Necessary upper-bound filter: identity cannot exceed min_len/max_len.
            max_len = max(len_i, len_j)
            if max_len == 0:
                possible_identity = 1.0
            else:
                possible_identity = min(len_i, len_j) / max_len
            if possible_identity + 1e-12 < identity_threshold:
                continue

            identity = _global_token_identity(
                encoded_sequences[sample_i],
                encoded_sequences[sample_j],
                aligner,
            )
            if identity + 1e-12 >= identity_threshold:
                union_find.union(sample_i, sample_j)

        if (sample_i + 1) % 100 == 0 or sample_i == n_samples - 2:
            print(f"  Compared sequences: {sample_i + 1}/{n_samples}")

    root_to_members: Dict[int, List[int]] = {}
    for sample_idx in range(n_samples):
        root = union_find.find(sample_idx)
        root_to_members.setdefault(root, []).append(sample_idx)

    # Stable cluster IDs: ordered by the first original row in each cluster.
    ordered_members = sorted(root_to_members.values(), key=lambda members: min(members))
    cluster_ids = np.empty(n_samples, dtype=int)
    clusters: Dict[int, np.ndarray] = {}
    for cluster_id, members in enumerate(ordered_members):
        member_array = np.asarray(members, dtype=int)
        clusters[cluster_id] = member_array
        cluster_ids[member_array] = cluster_id

    return cluster_ids, clusters


def _make_balance_features(y: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Build per-sample features used to balance cluster assignment.

    Features contain sample count, per-task valid-label counts, and per-task
    quantile-bin counts. They are used only for assigning whole clusters.
    """
    n_samples, n_tasks = y.shape
    valid_features = (~np.isnan(y)).astype(float)
    bin_features = np.zeros((n_samples, n_tasks * n_bins), dtype=float)

    for task_idx in range(n_tasks):
        values = y[:, task_idx]
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            continue

        quantiles = np.nanpercentile(
            values[valid_mask], np.linspace(0, 100, n_bins + 1)
        )
        inner_boundaries = quantiles[1:-1]
        task_bins = np.digitize(values[valid_mask], inner_boundaries)
        valid_indices = np.flatnonzero(valid_mask)
        for sample_idx, bin_idx in zip(valid_indices, task_bins):
            bin_features[sample_idx, task_idx * n_bins + int(bin_idx)] = 1.0

    sample_count_feature = np.ones((n_samples, 1), dtype=float)
    return np.concatenate(
        [sample_count_feature, valid_features, bin_features], axis=1
    )


def _assign_clusters_to_splits(
    clusters: Dict[int, np.ndarray],
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    n_bins: int,
) -> Dict[int, int]:
    """Greedily assign complete clusters while balancing size and labels."""
    if len(clusters) < 3:
        raise ValueError(
            "Sequence clustering produced fewer than three clusters, so a "
            "train/validation/test split is impossible. Try a higher identity threshold."
        )

    train_ratio = 1.0 - val_ratio - test_ratio
    split_ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    sample_features = _make_balance_features(y, n_bins=n_bins)

    cluster_features = {
        cluster_id: sample_features[members].sum(axis=0)
        for cluster_id, members in clusters.items()
    }
    total_features = sample_features.sum(axis=0)
    target_features = split_ratios[:, None] * total_features[None, :]
    feature_scale = np.maximum(total_features, 1.0)

    n_tasks = y.shape[1]
    feature_weights = np.concatenate(
        [
            np.asarray([4.0]),
            np.full(n_tasks, 1.0),
            np.full(n_tasks * n_bins, 0.25),
        ]
    )

    rng = np.random.default_rng(seed)
    cluster_order = list(clusters.keys())
    rng.shuffle(cluster_order)
    cluster_order.sort(key=lambda cid: len(clusters[cid]), reverse=True)

    current_features = np.zeros_like(target_features)
    cluster_to_split: Dict[int, int] = {}

    def assignment_loss(candidate_features: np.ndarray) -> float:
        squared_error = (
            (candidate_features - target_features) / feature_scale[None, :]
        ) ** 2
        return float(np.sum(squared_error * feature_weights[None, :]))

    for cluster_id in cluster_order:
        stats = cluster_features[cluster_id]
        candidate_splits = np.arange(3)
        rng.shuffle(candidate_splits)

        best_split = None
        best_loss = float("inf")
        for split_idx in candidate_splits:
            candidate = current_features.copy()
            candidate[split_idx] += stats
            loss = assignment_loss(candidate)
            if loss < best_loss:
                best_loss = loss
                best_split = int(split_idx)

        cluster_to_split[cluster_id] = best_split
        current_features[best_split] += stats

    used_splits = set(cluster_to_split.values())
    if used_splits != {0, 1, 2}:
        raise ValueError(
            "Cluster assignment left at least one split empty. Try a higher "
            "identity threshold or inspect the largest sequence clusters."
        )

    return cluster_to_split


def _build_ecfp_clusters(
    smiles: Sequence[str],
    similarity_threshold: float,
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> Tuple[np.ndarray, Dict[int, np.ndarray], List[str]]:
    """Build connected components from Backbone ECFP Tanimoto similarity."""
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in the interval (0, 1].")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")
    if len(smiles) == 0:
        raise ValueError("Cannot split an empty SMILES collection.")

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
    )
    canonical_smiles: List[str] = []
    structure_to_sample_indices: Dict[str, List[int]] = {}
    structure_to_fingerprint = {}

    for sample_idx, raw_smiles in enumerate(smiles):
        if not isinstance(raw_smiles, str) or not raw_smiles.strip():
            raise ValueError(f"Missing SMILES at row {sample_idx}.")
        molecule = Chem.MolFromSmiles(raw_smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {sample_idx}: {raw_smiles!r}")
        canonical = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        canonical_smiles.append(canonical)
        structure_to_sample_indices.setdefault(canonical, []).append(sample_idx)
        if canonical not in structure_to_fingerprint:
            structure_to_fingerprint[canonical] = fingerprint_generator.GetFingerprint(
                molecule
            )

    unique_structures = list(structure_to_sample_indices)
    fingerprints = [structure_to_fingerprint[smi] for smi in unique_structures]
    union_find = _UnionFind(len(unique_structures))
    print(
        f"Building ECFP{2 * radius} clusters for {len(smiles)} compounds "
        f"({len(unique_structures)} unique structures) at Tanimoto >= "
        f"{similarity_threshold:.2f} ..."
    )

    for structure_i in range(1, len(unique_structures)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[structure_i], fingerprints[:structure_i]
        )
        for structure_j, similarity in enumerate(similarities):
            if similarity + 1e-12 >= similarity_threshold:
                union_find.union(structure_i, structure_j)
        if structure_i % 250 == 0 or structure_i == len(unique_structures) - 1:
            print(
                f"  Compared unique structures: "
                f"{structure_i + 1}/{len(unique_structures)}"
            )

    root_to_sample_indices: Dict[int, List[int]] = {}
    for structure_idx, canonical in enumerate(unique_structures):
        root = union_find.find(structure_idx)
        root_to_sample_indices.setdefault(root, []).extend(
            structure_to_sample_indices[canonical]
        )

    ordered_members = sorted(
        root_to_sample_indices.values(), key=lambda members: min(members)
    )
    cluster_ids = np.empty(len(smiles), dtype=int)
    clusters: Dict[int, np.ndarray] = {}
    for cluster_id, members in enumerate(ordered_members):
        member_array = np.asarray(sorted(members), dtype=int)
        clusters[cluster_id] = member_array
        cluster_ids[member_array] = cluster_id

    return cluster_ids, clusters, canonical_smiles


def ecfp_cluster_split(
    smiles: Sequence[str],
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    similarity_threshold: float = 0.95,
    radius: int = 3,
    n_bits: int = 1024,
    use_chirality: bool = True,
    n_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Split Backbone compounds by ECFP/Tanimoto connected components."""
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must both be in (0, 1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if len(smiles) != y.shape[0]:
        raise ValueError("SMILES count and activity-array row count do not match.")

    cache_key = (
        float(similarity_threshold),
        int(radius),
        int(n_bits),
        bool(use_chirality),
        tuple(str(value) for value in smiles),
    )
    if cache_key in _ECFP_CLUSTER_CACHE:
        print("Reusing cached ECFP clusters from an earlier seed.")
        cluster_ids, clusters, canonical_smiles = _ECFP_CLUSTER_CACHE[cache_key]
    else:
        cluster_ids, clusters, canonical_smiles = _build_ecfp_clusters(
            smiles=smiles,
            similarity_threshold=similarity_threshold,
            radius=radius,
            n_bits=n_bits,
            use_chirality=use_chirality,
        )
        _ECFP_CLUSTER_CACHE[cache_key] = (
            cluster_ids,
            clusters,
            canonical_smiles,
        )

    cluster_to_split = _assign_clusters_to_splits(
        clusters=clusters,
        y=y,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        n_bins=n_bins,
    )
    sample_splits = np.asarray(
        [cluster_to_split[int(cluster_id)] for cluster_id in cluster_ids],
        dtype=int,
    )
    train_idx = np.flatnonzero(sample_splits == 0)
    val_idx = np.flatnonzero(sample_splits == 1)
    test_idx = np.flatnonzero(sample_splits == 2)

    cluster_sizes = sorted(
        (len(members) for members in clusters.values()), reverse=True
    )
    print(
        f"ECFP{2 * radius} clusters: {len(clusters)}; "
        f"largest cluster: {cluster_sizes[0]} compounds"
    )
    print(
        "ECFP cluster split sizes: "
        f"train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}"
    )

    return train_idx, val_idx, test_idx, cluster_ids, canonical_smiles


def sequence_cluster_split(
    sequences: Sequence[Sequence[str]],
    y: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    identity_threshold: float = 0.80,
    n_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split peptide data by token-aware sequence clusters.

    ncAA residue tokens are preserved exactly. Two sequences are connected when
    their Needleman-Wunsch global-alignment identity is at least the specified
    threshold. Each graph connected component is assigned wholly to one split.

    Returns
    -------
    train_idx, val_idx, test_idx, cluster_ids
        Row indices for each split plus a cluster ID for every input row.
    """
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must both be in (0, 1).")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if len(sequences) != y.shape[0]:
        raise ValueError("Sequence count and activity-array row count do not match.")

    # ===== reuse identical sequence clusters across multiple seeds =====
    cluster_cache_key = (
        float(identity_threshold),
        tuple(tuple(str(token) for token in sequence) for sequence in sequences),
    )
    if cluster_cache_key in _SEQUENCE_CLUSTER_CACHE:
        print("Reusing cached sequence clusters from an earlier seed.")
        cluster_ids, clusters = _SEQUENCE_CLUSTER_CACHE[cluster_cache_key]
    else:
        cluster_ids, clusters = _build_sequence_clusters(
            sequences=sequences,
            identity_threshold=identity_threshold,
        )
        _SEQUENCE_CLUSTER_CACHE[cluster_cache_key] = (cluster_ids, clusters)

    cluster_to_split = _assign_clusters_to_splits(
        clusters=clusters,
        y=y,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        n_bins=n_bins,
    )

    sample_splits = np.asarray(
        [cluster_to_split[int(cluster_id)] for cluster_id in cluster_ids],
        dtype=int,
    )
    train_idx = np.flatnonzero(sample_splits == 0)
    val_idx = np.flatnonzero(sample_splits == 1)
    test_idx = np.flatnonzero(sample_splits == 2)

    cluster_sizes = sorted(
        (len(members) for members in clusters.values()), reverse=True
    )
    print(
        f"Sequence clusters: {len(clusters)}; "
        f"largest cluster: {cluster_sizes[0]} samples"
    )
    print(
        "Cluster split sizes: "
        f"train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}"
    )

    return train_idx, val_idx, test_idx, cluster_ids
