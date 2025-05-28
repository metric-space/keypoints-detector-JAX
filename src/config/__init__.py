from dataclasses import dataclass


@dataclass
class Config:
    max_samples: int = 18000
    dataset_directory: str = "./data"

    batch_size: int = 30
    image_size: tuple[int, int] = (64, 64)

    lr: float = 4e-4
    lr_decay_steps: int = 100
    lr_alpha: float = 0.2

    input_channels: int = 3
    output_channels: int = 5
    max_features: int = 256
    num_blocks: int = 5
    block_expansion: int = 32

    data_seed: int = 3728
    nn_seed: int = 1023

    steps: int = 100
