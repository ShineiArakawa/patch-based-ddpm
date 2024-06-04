import argparse

import accelerate
import loguru

import patch_based_ddpm.denoising_diffusion_pytorch as ddpm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Patch-based Denoising Diffusion Probabilistic Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config file"
    )

    parser.add_argument(
        "--restart-step",
        type=int,
        default=1,
        help="Restart training from the given step"
    )

    return parser.parse_args()


def main():
    logger = loguru.logger

    args = parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config: ddpm.Config = ddpm.Config.load(args.config)

    # Create model
    model = ddpm.patched_unet.PatchedUnetWithGlobalEncoding(config.model)

    # Create diffusion model
    diffusion_model = ddpm.diffusion.GaussianDiffusion(config, model)

    # Create trainer
    trainer = ddpm.trainer.Trainer(diffusion_model, config)

    if args.restart_step > 1:
        trainer.load(args.restart_step)

    trainer.train()

    logger.info("Bye.")


if __name__ == "__main__":
    main()
