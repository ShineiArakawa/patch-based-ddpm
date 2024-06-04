from __future__ import annotations

import inspect
import typing

import torch
import torch.utils.data as torch_data

# helpers functions


def exists(x: typing.Any):
    return x is not None


def default(val: typing.Any, d: typing.Any):
    if exists(val):
        return val
    return d() if inspect.isfunction(d) else d


def cycle(dl: torch_data.DataLoader):
    while True:
        for data in dl:
            yield data


def num_to_groups(num: int, divisor: int) -> list[int]:
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def normalize_to_neg_one_to_one(img: torch.Tensor) -> torch.Tensor:
    return img * 2.0 - 1.0


def unnormalize_to_zero_to_one(t: torch.Tensor) -> torch.Tensor:
    return (t + 1.0) * 0.5
