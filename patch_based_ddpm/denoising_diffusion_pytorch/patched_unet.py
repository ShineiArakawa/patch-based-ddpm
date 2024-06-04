from functools import partial

import torch
import torch.nn as nn

from patch_based_ddpm.denoising_diffusion_pytorch import config as cnf
from patch_based_ddpm.denoising_diffusion_pytorch import modules, utils


class Embedding(nn.Module):
    def __init__(self, dim: int, patch_divider: int) -> None:
        """Embed diffusion time step and position into a single tensor.

        Args:
            dim (int): internal dimension
            patch_divider (int): the number of patches in one dimension
        """

        super().__init__()

        self.dim = dim
        self.patch_divider = patch_divider

        embedding_dim = dim * 4
        self._time_embedding_layer = nn.Sequential(
            modules.SinusoidalPosEmb(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )
        self._pos_embedding_layer = nn.Sequential(
            nn.Linear(self.patch_divider * self.patch_divider, self.dim),
            nn.GELU()
        )
        self._mapping = nn.Sequential(
            nn.Linear(2 * dim, embedding_dim)
        )

    def forward(self, time: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        time_embedding: torch.Tensor = self._time_embedding_layer(time)

        tiled_time_embed = time_embedding.tile((self.patch_divider * self.patch_divider, 1))

        pos_embedding: torch.Tensor = self._pos_embedding_layer(pos)

        return self._mapping(torch.cat((tiled_time_embed, pos_embedding), dim=-1))


class PatchedEncoder(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        init_dim: int = None,
        dim: int = 64,
        dim_mults: tuple[int] = (1, 2, 4),
        resnet_block_groups: int = 8
    ) -> None:
        super().__init__()

        init_dim = utils.default(init_dim, dim // 3 * 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        embedDim = dim * 4
        num_resolutions = len(in_out)
        block_class = partial(modules.ResnetBlock, groups=resnet_block_groups)

        self._init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        self._downsLayers = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self._downsLayers.append(nn.ModuleList([
                block_class(dim_in, dim_out, time_emb_dim=embedDim),
                block_class(dim_out, dim_out, time_emb_dim=embedDim),
                modules.Residual(modules.PreNorm(dim_out, modules.LinearAttention(dim_out))),
                modules.Downsample(dim_out) if not is_last else nn.Identity()
            ]))

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self._init_conv(x)

        h = []

        for i, (block1, block2, attn, downsample) in enumerate(self._downsLayers):
            x = block1(x, embedding)
            x = block2(x, embedding)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        return x, h


class Decoder(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        dim_mults: list[int] = (1, 2, 4),
        init_dim: int = None,
        out_dim: int = None,
        channels: int = 3,
        resnet_block_groups: int = 8,
        learned_variance: bool = False
    ) -> None:
        super().__init__()

        init_dim = utils.default(init_dim, dim // 3 * 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        mid_dim = dims[-1]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_resolutions = len(in_out)
        embedDim = dim * 4
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = utils.default(out_dim, default_out_dim)
        block_class = partial(modules.ResnetBlock, groups=resnet_block_groups)

        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim=embedDim)
        self.mid_attn = modules.Residual(modules.PreNorm(mid_dim, modules.Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim=embedDim)

        self._embedder = nn.Linear(dim, embedDim)
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_class(dim_out * 2, dim_in, time_emb_dim=embedDim),
                block_class(dim_in, dim_in, time_emb_dim=embedDim),
                modules.Residual(modules.PreNorm(dim_in, modules.LinearAttention(dim_in))),
                modules.Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            block_class(dim, dim),
            nn.Conv2d(dim, self.out_dim, 1)
        )

    def forward(self, x: torch.Tensor, h: list[torch.Tensor], embedding: torch.Tensor) -> torch.Tensor:
        x = self.mid_block1(x, embedding)
        x = self.mid_attn(x)
        x = self.mid_block2(x, embedding)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, embedding)
            x = block2(x, embedding)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


class PatchedUnetWithGlobalEncoding(nn.Module):
    def __init__(self, config: cnf.ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.patch_size: tuple[int, int] = (config.image_size // config.patch_divider, config.image_size // config.patch_divider)

        self._pooling = nn.AdaptiveAvgPool2d(output_size=self.patch_size)

        self._patched_encoder = PatchedEncoder(
            channels=config.channels * 2,  # NOTE: This is because we concatenate the global attribute to the patch
            init_dim=config.init_dim,
            dim=config.dim,
            dim_mults=config.dim_mults,
            resnet_block_groups=config.resnet_block_groups
        )

        self._patched_decoder = Decoder(
            dim=config.dim,
            dim_mults=config.dim_mults,
            init_dim=config.init_dim,
            out_dim=config.out_dim,
            channels=config.channels,
            resnet_block_groups=config.resnet_block_groups,
            learned_variance=config.learned_variance
        )

        self._embedding = Embedding(
            dim=config.dim,
            patch_divider=config.patch_divider
        )

    def _make_patch_with_global_attribute(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patches from the input image and concatenate global attribute to each patch.

        Args:
            x (torch.Tensor): input image

        Returns:
            tuple[torch.Tensor, torch.Tensor]: patched images and position 
        """
        b, _, h, w = x.shape

        ls_patchs: list[torch.Tensor] = []
        ls_pos: list[torch.Tensor] = []

        # Extract global attribute
        globalAttr: torch.Tensor = self._pooling(x)

        for i in range(self.config.patch_divider):
            for j in range(self.config.patch_divider):
                index = i * self.config.patch_divider + j

                # Extract patch
                patch = x[
                    :,
                    :,
                    h // self.config.patch_divider * i: w // self.config.patch_divider * (i + 1),
                    h // self.config.patch_divider * j: w // self.config.patch_divider * (j + 1)
                ]

                patch = torch.cat((patch, globalAttr), dim=1)

                pos = torch.zeros((b, self.config.patch_divider * self.config.patch_divider), device=x.device)
                pos[:, index] = 1

                ls_patchs.append(patch)
                ls_pos.append(pos)

        output = torch.cat(ls_patchs)
        position = torch.cat(ls_pos)

        return output, position

    def _rearrange_patch(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange patches to the entire image.

        Args:
            x (torch.Tensor): patched image

        Returns:
            torch.Tensor: entire image
        """
        b_, c, h, w = x.shape
        b = b_ // (self.config.patch_divider * self.config.patch_divider)

        output = torch.empty(
            size=(
                b,
                c,
                h * self.config.patch_divider,
                w * self.config.patch_divider
            ),
            device=x.device
        )

        for i in range(self.config.patch_divider):
            for j in range(self.config.patch_divider):
                index = i * self.config.patch_divider + j

                patch = x[index * b: (index + 1) * b, :, :, :]
                output[:, :, h * i:w * (i + 1), h * j:w * (j + 1)] = patch

        return output

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, p = self._make_patch_with_global_attribute(x)

        embedding = self._embedding(t, p)

        x, h = self._patched_encoder(x, embedding)
        x = self._patched_decoder(x, h, embedding)

        x = self._rearrange_patch(x)

        return x
