from __future__ import annotations

import copy
import math
import pathlib
import typing

import accelerate
import loguru
import PIL.Image as Image
import torch
import torch.amp
import torch.optim as torch_optim
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.utils as torchvision_utils
import tqdm
import wandb

# Add the missing import statement
from patch_based_ddpm.denoising_diffusion_pytorch import config as cnf
from patch_based_ddpm.denoising_diffusion_pytorch import (diffusion, modules,
                                                          utils)


class Dataset(torch_data.Dataset):
    def __init__(self, folder: pathlib.Path, image_size: tuple[int, int], exts: list[str] = ['jpg', 'jpeg', 'png']):
        super().__init__()

        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in self.folder.glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class SampleDataset(torch_data.Dataset):
    def __init__(self, num_samples: int):
        super().__init__()

        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> list[pathlib.Path]:
        return index


class Trainer(object):
    def __init__(
        self,
        diffusion_model: diffusion.GaussianDiffusion,
        config: cnf.Config
    ):
        super().__init__()

        self.logger = loguru.logger
        self.trainer_config: cnf.TrainerConfig = config.trainer
        self.model: diffusion.GaussianDiffusion = diffusion_model
        self.step: int = 0

        # ============================================================================================================
        # Initialize Accelerator
        # ============================================================================================================
        self.accelerator: accelerate.Accelerator = accelerate.Accelerator(
            kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)],
            mixed_precision='fp16' if self.trainer_config.use_amp else 'no'
        )

        # Set seed
        self.logger.info(f"Set seed to {config.trainer.seed}")
        accelerate.utils.set_seed(self.trainer_config.seed, device_specific=True)

        # ============================================================================================================
        # Initialize dataloader
        # ============================================================================================================
        self.ds = Dataset(self.trainer_config.train_dataset_dir, config.model.image_size)

        dl = torch_data.DataLoader(
            self.ds,
            batch_size=self.trainer_config.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.trainer_config.num_workers
        )

        dl = self.accelerator.prepare(dl)

        self.dl = utils.cycle(dl)

        # Dataloader for sample
        sample_num_batches_per_device = math.ceil(36 / self.accelerator.num_processes)
        self.sample_ds = torch_data.TensorDataset(torch.zeros(36))

        sample_dl = torch_data.DataLoader(
            self.sample_ds,
            batch_size=sample_num_batches_per_device
        )

        self.sample_dl = self.accelerator.prepare(sample_dl)

        # ============================================================================================================
        # Initialize accelerate model and optimizer
        # ============================================================================================================
        self.opt = torch_optim.Adam(self.model.parameters(), lr=self.trainer_config.train_lr)

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # ============================================================================================================
        # Initialize wandb
        # ============================================================================================================
        if self.accelerator.is_main_process and self.trainer_config.wandb_enabled:
            wandb.init(
                project=self.trainer_config.wandb_project,
                dir=self.trainer_config.wandb_log_dir,
                name=self.trainer_config.wandb_name,
                config=config.to_dict(),
            )

            wandb.watch(self.model)

        # ============================================================================================================
        # Initialize EMA model
        # ============================================================================================================
        self.ema = modules.EMA(self.trainer_config.ema_decay)
        self.ema_model: diffusion.GaussianDiffusion = copy.deepcopy(self.accelerator.unwrap_model(self.model))
        self.reset_parameters()

    @property
    def device(self):
        return self.accelerator.device

    def reset_parameters(self) -> None:
        self.ema_model.load_state_dict(self.accelerator.get_state_dict(self.model))

    def step_ema(self) -> None:
        if self.step < self.trainer_config.step_start_ema:
            self.reset_parameters()
            return

        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, file_name: str) -> None:
        if self.accelerator.is_main_process:
            checkpoint = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt': self.opt.state_dict(),
                'ema': self.ema_model.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if utils.exists(self.accelerator.scaler) else None,
            }

            file_path = self.trainer_config.checkpoints_dir / file_name
            torch.save(checkpoint, file_path)

            self.logger.info(f"saved checkpoint to {file_path}")

    def load(self, milestone: int) -> None:
        checkpoint_path = self.trainer_config.checkpoints_dir / f'model-{milestone}.pt'
        checkpoint_path = checkpoint_path.absolute()

        self.load_from_file(file_path=checkpoint_path)

    def load_from_file(self, file_path: pathlib.Path) -> None:
        self.logger.info(f"loading from {file_path}")

        checkpoint = torch.load(
            file_path,
            map_location=self.device
        )

        # Load model
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint['model'])
        self.model = self.accelerator.prepare_model(model)

        # Load step
        self.step = checkpoint['step']

        # Load optimizer
        self.opt.load_state_dict(checkpoint['opt'])

        # Load EMA
        self.ema_model.load_state_dict(checkpoint['ema'])

        # Load scaler
        if utils.exists(self.accelerator.scaler) and utils.exists(checkpoint['scaler']):
            self.accelerator.scaler.load_state_dict(checkpoint['scaler'])

    def train(self) -> None:
        with tqdm.tqdm(
            initial=self.step,
            total=self.trainer_config.train_num_steps,
            disable=not self.accelerator.is_main_process
        ) as pbar:
            while self.step < self.trainer_config.train_num_steps:
                total_loss: float = 0.0

                # ============================================================================================================
                # Backprop and update
                # ============================================================================================================
                for accumulation_step in range(self.trainer_config.gradient_accumulate_every):
                    data: torch.Tensor = next(self.dl).to(self.device)

                    with self.accelerator.autocast():
                        loss: torch.Tensor = self.model(data)
                        loss = loss / self.trainer_config.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.accelerator.is_main_process and self.trainer_config.wandb_enabled:
                    wandb.log({'loss': total_loss}, step=self.step)

                self.opt.step()
                self.opt.zero_grad()

                if self.step % self.trainer_config.update_ema_every == 0:
                    # ============================================================================================================
                    # Update EMA
                    # ============================================================================================================
                    self.accelerator.wait_for_everyone()

                    self.step_ema()

                if self.step != 0 and self.step % self.trainer_config.save_and_sample_every == 0:
                    self.accelerator.wait_for_everyone()

                    # ============================================================================================================
                    # Unconditional sample
                    # ============================================================================================================
                    self.ema_model.eval()

                    with torch.inference_mode():
                        for batch in self.sample_dl:
                            batch_size = batch[0].shape[0]

                            with self.accelerator.autocast():
                                sampled_image = self.ema_model.sample(batch_size=batch_size, rank=self.accelerator.process_index)

                            all_images = self.accelerator.gather(sampled_image)[:36]

                            if self.accelerator.is_main_process:
                                all_images = torchvision_utils.make_grid(all_images, nrow=6)

                                pil_image: Image.Image = F.to_pil_image(all_images)
                                pil_image.save(self.trainer_config.sampled_dir / f'sample-{self.step}.png')

                                if self.trainer_config.wandb_enabled:
                                    wandb.log({'sample': [wandb.Image(pil_image)]}, step=self.step)

                    if self.accelerator.is_main_process:
                        # ============================================================================================================
                        # Save checkpoint
                        # ============================================================================================================
                        self.save(f'model-{self.step}.pt')

                    self.accelerator.wait_for_everyone()

                self.step += 1
                pbar.update(1)

        self.save('model-last.pt')
        self.logger.info('training complete')

    def sample(
        self,
        num_samples: int,
        batch_size: int,
        output_dir: pathlib.Path
    ) -> None:
        self.logger.info(f"Sampling {num_samples} images to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        sample_dataset = SampleDataset(num_samples)

        sample_dataloader = torch_data.DataLoader(
            sample_dataset,
            batch_size=batch_size
        )
        sample_dataloader = self.accelerator.prepare(sample_dataloader)

        self.ema_model.eval()

        with torch.inference_mode():
            for batch in tqdm.tqdm(sample_dataloader, disable=not self.accelerator.is_main_process, desc='sampling'):
                global_indices: list[int] = batch
                batch_size = len(global_indices)

                with self.accelerator.autocast():
                    sampled_image = self.ema_model.sample(batch_size=batch_size, rank=self.accelerator.process_index, disable_pbar=True)

                for i, image in enumerate(sampled_image):
                    image = F.to_pil_image(image)

                    out_file_path = output_dir / f'sample-{global_indices[i]:08}.png'

                    image.save(out_file_path)
