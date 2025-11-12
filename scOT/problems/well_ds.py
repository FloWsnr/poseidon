"""
Wrapper around the WellDataset to provide additional functionality.
By Florian Wiesner
Date: 2025-05-05
"""

from typing import Optional, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

from the_well.data.datasets import WellDataset
from the_well.data.normalization import ZScoreNormalization


class PhysicsDataset(WellDataset):
    """Wrapper around the WellDataset.

    Returns a dictionary with keys:
        - "pixel_values": input tensor of shape (c, h, w)
        - "labels": label tensor of shape (c, h, w)
        - "time": time value (currently set to 1)
        - "pixel_mask": mask tensor of shape (c,)

    Parameters
    ----------
    data_dir : Path
        Path to the data directory (e.g. "data/physics_data/train")
    use_normalization: bool
        Whether to use normalization
        By default False
    dt_stride: int
        Time step stride between samples
        By default 1
    full_trajectory_mode: bool
        Whether to use the full trajectory mode of the well dataset.
        This returns full trajectories instead of individual timesteps.
        By default False
    nan_to_zero: bool
        Whether to replace NaNs with 0
        By default True
    num_channels: int
        Number of channels in the data
        By default 5
    """

    def __init__(
        self,
        data_dir: Path,
        use_normalization: bool = True,
        dt_stride: int | list[int] = 1,
        full_trajectory_mode: bool = False,
        nan_to_zero: bool = True,
        num_channels: int = 5,
    ):
        if isinstance(dt_stride, list):
            min_dt_stride = dt_stride[0]
            max_dt_stride = dt_stride[1]
        else:
            min_dt_stride = dt_stride
            max_dt_stride = dt_stride

        super().__init__(
            path=str(data_dir),
            normalization_path=str(data_dir.parents[1] / "stats.yaml"),
            n_steps_input=1,
            n_steps_output=1,
            use_normalization=use_normalization,
            normalization_type=ZScoreNormalization,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            full_trajectory_mode=full_trajectory_mode,
        )
        self.nan_to_zero = nan_to_zero
        # give the dataset its correct name
        name = data_dir.parents[1].name
        self.dataset_name = name

        self.pixel_mask = torch.tensor([False] * num_channels)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index) -> dict[str, Any]:
        data = super().__getitem__(index)  # returns (1, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]

        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
        # reshape to (c, h, w)
        x = rearrange(x, "1 h w c -> 1 c h w")
        y = rearrange(y, "1 h w c -> 1 c h w")

        # interpolate to 128x128
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)

        # squeeze the batch dimension
        x = x.squeeze(0)
        y = y.squeeze(0)

        return {
            "pixel_values": x,
            "labels": y,
            "time": 1,
            "pixel_mask": self.pixel_mask,
        }
