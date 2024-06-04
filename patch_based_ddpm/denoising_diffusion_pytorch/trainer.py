from __future__ import annotations

import copy
import pathlib

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


class Trainer(object):
    def __init__(
        self,
        diffusion_model: diffusion.GaussianDiffusion,
        config: cnf.Config
    ):
        super().__init__()

        self.logger = loguru.logger
        self.trainer_config: cnf.TrainerConfig = config.trainer

        self.accelerator = accelerate.Accelerator(
            split_batches=True,
            mixed_precision='fp16' if self.trainer_config.use_amp else 'no'
        )

        self.model = diffusion_model

        self.step = 0

        # ============================================================================================================
        # Initialize EMA model
        # ============================================================================================================
        if self.accelerator.is_main_process:
            self.ema = modules.EMA(self.trainer_config.ema_decay)
            self.ema_model = copy.deepcopy(self.model)
            self.reset_parameters()

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

    @property
    def device(self):
        return self.accelerator.device

    def reset_parameters(self) -> None:
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self) -> None:
        if self.accelerator.is_main_process:
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

        self.logger.info(f"loading from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
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

        if self.accelerator.is_main_process:
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

                if self.accelerator.is_main_process and self.step != 0 and self.step % self.trainer_config.save_and_sample_every == 0:
                    # ============================================================================================================
                    # Unconditional sample
                    # ============================================================================================================
                    self.accelerator.wait_for_everyone()
                    self.ema_model.eval()

                    with torch.inference_mode():
                        batches = utils.num_to_groups(36, self.trainer_config.train_batch_size)
                        all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))

                    all_images = torchvision_utils.make_grid(torch.cat(all_images_list, dim=0), nrow=6)

                    pil_image: Image.Image = F.to_pil_image(utils.unnormalize_to_zero_to_one(all_images))
                    pil_image.save(
                        self.trainer_config.sampled_dir / f'sample-{self.step}.png'
                    )

                    if self.trainer_config.wandb_enabled:
                        wandb.log({'sample': [wandb.Image(pil_image)]}, step=self.step)

                    # ============================================================================================================
                    # Save checkpoint
                    # ============================================================================================================
                    self.save(f'model-{self.step}.pt')

                self.step += 1
                pbar.update(1)

        self.save('model-last.pt')
        self.logger.info('training complete')
