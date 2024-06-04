from __future__ import annotations

import dataclasses
import os
import pathlib

import dataclasses_json

from patch_based_ddpm.denoising_diffusion_pytorch import utils


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class TrainerConfig:
    train_dataset_dir: pathlib.Path
    log_dir: pathlib.Path

    ema_decay: float = 0.995
    train_batch_size: int = 32
    train_lr: float = 1e-4
    train_num_steps: int = 100000
    gradient_accumulate_every: int = 1
    use_amp: bool = False
    step_start_ema: int = 2000
    update_ema_every: int = 10
    save_and_sample_every: int = 1000
    num_workers: int = os.cpu_count()

    wandb_enabled: bool = False
    wandb_project: str | None = None
    wandb_name: str | None = None

    @property
    def checkpoints_dir(self) -> pathlib.Path:
        dir_path = self.log_dir / 'checkpoints'
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def sampled_dir(self) -> pathlib.Path:
        dir_path = self.log_dir / 'sampled'
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @property
    def wandb_log_dir(self) -> pathlib.Path:
        dir_path = self.log_dir / 'wandb'
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def __post_init__(self):
        self.train_dataset_dir = pathlib.Path(self.train_dataset_dir)
        self.log_dir = pathlib.Path(self.log_dir)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class ModelConfig:
    dim: int
    init_dim: int | None = None
    out_dim: int | None = None
    dim_mults: list[int] = (1, 2, 4, 8)
    channels: int = 3
    resnet_block_groups: int = 8
    learned_variance: bool = False
    patch_divider: int = 4
    image_size: int = 128

    def __post_init__(self):
        default_out_dim = self.channels * (1 if not self.learned_variance else 2)
        self.out_dim = utils.default(self.out_dim, default_out_dim)


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class DiffusionConfig:
    timesteps: int = 1000
    loss_type: str = 'l1'
    objective: str = 'pred_noise'


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class Config:
    trainer: TrainerConfig
    model: ModelConfig
    diffusion: DiffusionConfig

    @staticmethod
    def load(file_path: str, encoding: str = "utf-8") -> Config:
        """Load configs from the specified json file and override initial configs

        Parameters
        ----------
        file_path : str
            path to the config file
        encoding : str, optional
            file encoding, by default "utf-8"

        Returns
        -------
        Config
            overrided config
        """
        config: Config = None

        with open(file_path, encoding=encoding) as file:
            config = Config.from_json(file.read())
            pass

        return config
