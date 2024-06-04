from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from patch_based_ddpm.denoising_diffusion_pytorch import config as cnf
from patch_based_ddpm.denoising_diffusion_pytorch import patched_unet, utils

# small helper modules


# gaussian diffusion trainer class
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Tensor) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape: tuple[int, ...], device: str | torch.device, repeat: bool = False) -> torch.Tensor:
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1

    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        config: cnf.Config,
        denoise_fn: patched_unet.PatchedUnetWithGlobalEncoding
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and denoise_fn.config.channels != denoise_fn.config.out_dim)

        self.config = config
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(self.config.diffusion.timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = config.diffusion.loss_type

        # helper function to register buffer from float64 to float32
        def register_buffer(name: str, val: torch.Tensor) -> torch.Tensor:
            return self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        pass

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Calculate x_{t=0} from x_t and noise

        Args:
            x_t (torch.Tensor): Noisy image at time t
            t (torch.Tensor): time step
            noise (torch.Tensor): noise

        Returns:
            torch.Tensor: x_{t=0}
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate posterior q(x_{t-1} | x_t, x_0) according to the equation (6)/(7) in the DDPM paper

        Args:
            x_start (torch.Tensor): x_{t=0}
            x_t (torch.Tensor): x_t
            t (torch.Tensor): time step

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: posterior mean, posterior variance, posterior log variance clipped
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  # equation the (7) in DDPM paper

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Calculate mean and variance of the model p_{\theta}(x_{t-1} | x_t) according to the equation (11) in the DDPM paper

        Args:
            x (torch.Tensor): x_t
            t (torch.Tensor): time step
            clip_denoised (bool): whether to clip the denoised image

        Raises:
            ValueError: unknown objective

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: model mean, posterior variance, posterior log variance
        """
        model_output = self.denoise_fn(x, t)

        if self.config.diffusion.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.config.diffusion.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.config.diffusion.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True, repeat_noise: bool = False) -> torch.Tensor:
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)

        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple[int, ...], rank: int = None, disable_pbar: bool = False) -> torch.Tensor:
        """Run the reverse diffusion process

        Args:
            shape (tuple[int, ...]): input shape

        Returns:
            torch.Tensor: sampled image
        """
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        tqdm_label = 'sampling loop time step' if rank is None else f'sampling loop time step (rank: {rank})'

        for i in tqdm.tqdm(reversed(range(0, self.num_timesteps)), desc=tqdm_label, total=self.num_timesteps, disable=disable_pbar):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        img = utils.unnormalize_to_zero_to_one(img)

        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16, rank: int = None, disable_pbar: bool = False) -> torch.Tensor:
        image_size = self.config.model.image_size
        channels = self.config.model.channels

        return self.p_sample_loop((batch_size, channels, image_size, image_size), rank, disable_pbar)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Calculate x_t from x_{t=0} and noise according to the equation (4) in the DDPM paper

        Args:
            x_start (torch.Tensor): x_{t=0}
            t (torch.Tensor): time step
            noise (torch.Tensor, optional): Gaussian noise. Defaults to None.

        Returns:
            torch.Tensor: x_t
        """

        noise = utils.default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        noise = utils.default(noise, lambda: torch.randn_like(x_start))

        # NOTE: calculate x_t from x_{t=0} and noise
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.denoise_fn(x, t)

        if self.config.diffusion.objective == 'pred_noise':
            target = noise
        elif self.config.diffusion.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective { self.config.diffusion.objective}')

        loss = self.loss_fn(model_out, target)

        return loss

    def forward(self, img: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Calculate loss

        Args:
            img (torch.Tensor): input image

        Returns:
            torch.Tensor: loss
        """

        b, c, h, w, device, img_size, = *img.shape, img.device, self.config.model.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = utils.normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, *args, **kwargs)

# dataset classes
