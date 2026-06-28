from pydantic import BaseModel


class SamplingConfig(BaseModel):
    model_type: str
    model_path: str
    results_output: str
    input_sequences_path: str = ''
    batch_size: int = 1
    num_samples: int = 1000
    beam_size: int = 1000
    # decoding strategy: 'multinomial' (default), 'greedy', or 'beamsearch'
    decode_type: str = "multinomial"
    temperature: float = 1.0
    # performance knobs
    num_workers: int = 0
    sequences_per_chunk: int = 64
    show_progress: bool = True
    run_type: str = 'sampling'
