from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
    MultiUpscaler,
    UpscalerCheckpoints,
)

from esrgan_model import UpscalerESRGAN


@dataclass(kw_only=True)
class ESRGANUpscalerCheckpoints(UpscalerCheckpoints):
    esrgan: Path


class ESRGANUpscaler(MultiUpscaler):
    def __init__(
        self,
        checkpoints: ESRGANUpscalerCheckpoints,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(checkpoints=checkpoints, device=device, dtype=dtype)
        self.esrgan = UpscalerESRGAN(checkpoints.esrgan, device=self.device, dtype=self.dtype)

    def to(self, device: torch.device, dtype: torch.dtype):
        self.esrgan.to(device=device, dtype=dtype)
        self.sd = self.sd.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def pre_upscale(self, image: Image.Image, upscale_factor: float, **_: Any) -> Image.Image:
        image = self.esrgan.upscale_with_tiling(image)
        return super().pre_upscale(image=image, upscale_factor=upscale_factor / 4)
