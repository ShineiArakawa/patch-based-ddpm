import argparse
import pathlib

import accelerate
import loguru
import torch

import patch_based_ddpm.denoising_diffusion_pytorch as ddpm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample images from a Patch-based Denoising Diffusion Probabilistic Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the config file"
    )

    parser.add_argument(
        "--ckpt",
        type=pathlib.Path,
        required=True,
        help="Path to the checkpoint file"
    )

    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the checkpoint file"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples to generate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sampling"
    )

    return parser.parse_args()


def main():
    logger = loguru.logger

    args = parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config: ddpm.Config = ddpm.Config.load(args.config)

    # NOTE: Turn off wandb
    config.trainer.wandb_enabled = False

    # Create model
    model = ddpm.patched_unet.PatchedUnetWithGlobalEncoding(config.model)

    # Create diffusion model
    diffusion_model = ddpm.diffusion.GaussianDiffusion(config, model)

    # Create trainer
    trainer = ddpm.trainer.Trainer(diffusion_model, config)
    trainer.load_from_file(args.ckpt)

    # Sample
    trainer.sample(args.n_samples, args.batch_size, args.out_dir)

    logger.info("Bye.")


if __name__ == "__main__":
    main()
