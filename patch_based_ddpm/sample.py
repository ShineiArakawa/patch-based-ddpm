from patchedUnet import PatchedUnetWithGlobalEncoding
from denoising_diffusion_pytorch import GaussianDiffusion
from setting import Setting

import torch
from torch.nn.parallel import DataParallel
import numpy as np
import os
import sys
import argparse
from multiprocessing import Pool
from PIL import Image


class CustomParallel(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("pathTrainedParam")
    parser.add_argument("pathSaveDir")
    parser.add_argument("settingFilePath")
    parser.add_argument("--jobID", default=0, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_iter", default=200, type=int)
    parser.add_argument("--nParallels", default=4, type=int)
    args = parser.parse_args()

    pathTrainedParam = args.pathTrainedParam
    pathSaveDir = args.pathSaveDir
    settingFilePath = args.settingFilePath
    jobID = args.jobID
    batch_size = args.batch_size
    num_iter = args.num_iter
    nParallels = args.nParallels

    setting = Setting(settingFilePath)
    dim = setting.dim
    patchDivider = setting.patchDivider
    dim_mults = setting.dim_mults
    timeSteps = setting.timeSteps
    imageSize = setting.imageSize

    model = PatchedUnetWithGlobalEncoding(
        dim=dim,
        dim_mults=dim_mults,
        patchDivider=patchDivider
    )
    state_dict = torch.load(
        pathTrainedParam,
        map_location=torch.device("cpu")
    )

    model = CustomParallel(model)

    diffusion = GaussianDiffusion(
        model,
        image_size=imageSize,
        timesteps=timeSteps,
        loss_type='l1'
    )
    diffusion.load_state_dict(state_dict["ema"])

    diffusion = diffusion.cuda()

    os.makedirs(pathSaveDir, exist_ok=True)
    for i in range(num_iter):
        print(
            f"Sampling  |  {i+1}/{num_iter} ==================================================="
        )
        sampledImage: torch.Tensor = diffusion.sample(batch_size=batch_size)

        sampledImage: np.ndarray = sampledImage.detach().clone().permute(
            0,
            2,
            3,
            1
        ).cpu().numpy()

        ls_args = []
        for j in range(batch_size):
            image = sampledImage[j]
            pathImage = os.path.join(
                pathSaveDir,
                f"image_{jobID}_{i * batch_size + j}.png"
            )
            args = (image, pathImage)
            ls_args.append(args)

        with Pool(processes=nParallels) as procecss:
            procecss.map(func=saveImages_wapped, iterable=ls_args)


def saveImages_wapped(args):
    saveImages(*args)
    pass


def saveImages(imageArray: np.ndarray, pathImage: str):
    image = Image.fromarray((imageArray*255).astype(np.uint8))
    image.save(pathImage)


if __name__ == "__main__":
    args = sys.argv
    main(args)
