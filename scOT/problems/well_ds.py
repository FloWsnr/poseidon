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


from gphyt.data.well_dataset import WellDataset, ZScoreNormalization


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
            normalization_path=str(data_dir.parents[0] / "stats.yaml"),
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
        self.max_time = self.metadata.n_steps_per_trajectory[0] - 1

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index) -> dict[str, Any]:
        data, metadata = super().__getitem__(index)  # returns (1, h, w, c)
        x = data["input_fields"]
        y = data["output_fields"]

        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
        # reshape to (c, h, w)
        x = rearrange(x, "1 h w c -> 1 c h w")
        y = rearrange(y, "1 h w c -> 1 c h w")

        # # interpolate to 128x128
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(128, 128), mode="bilinear", align_corners=False)

        # squeeze the batch dimension
        x = x.squeeze(0)
        y = y.squeeze(0)

        dt = metadata.time_stride

        return {
            "pixel_values": x,
            "labels": y,
            "time": torch.tensor(dt),
            "pixel_mask": self.pixel_mask,
        }


class SuperDataset:
    """Wrapper around a list of datasets.

    Allows to concatenate multiple datasets and randomly sample from them.

    Parameters
    ----------
    datasets : dict[str, PhysicsDataset]
        Dictionary of datasets to concatenate
    out_shape : tuple[int, int]
        Output shape (h, w) of the concatenated dataset.
        This is needed to account for the different shapes of the datasets.
    max_samples_per_ds : Optional[int | list[int]]
        Maximum number of samples to sample from each dataset.
        If a list, specifies the number of samples for each dataset individually.
        If None, uses all samples from each dataset.
        By default None.

    return_ds_idx : bool
        Whether to return the dataset index along with the data.
        This is used for PINN losses to know which dataset the sample comes from.
        By default False.

    seed : Optional[int]
        Random seed for reproducibility.
        By default None.
    """

    def __init__(
        self,
        datasets: dict[str, PhysicsDataset],
        max_samples_per_ds: Optional[int | list[int]] = None,
        seed: Optional[int] = None,
        return_ds_idx: bool = False,
    ):
        self.datasets = datasets
        self.dataset_list = list(datasets.values())
        self.return_ds_idx = return_ds_idx

        if isinstance(max_samples_per_ds, int):
            self.max_samples_per_ds = [max_samples_per_ds] * len(datasets)
        else:
            self.max_samples_per_ds = max_samples_per_ds

        self.seed = seed

        # Initialize random number generator
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

        # Generate initial random indices
        self.reshuffle()

    def reshuffle(self):
        """Reshuffle the indices for each dataset.

        This should be called at the start of each epoch to ensure
        a new random subset of samples is used.

        """
        self.dataset_indices = []
        for i, dataset in enumerate(self.dataset_list):
            if (
                self.max_samples_per_ds is not None
                and len(dataset) > self.max_samples_per_ds[i]
            ):
                indices = torch.randperm(len(dataset), generator=self.rng)[
                    : self.max_samples_per_ds[i]
                ]
                self.dataset_indices.append(indices)
            else:
                self.dataset_indices.append(None)

        # Calculate lengths based on either max_samples_per_ds or full dataset length
        self.lengths = [
            min(self.max_samples_per_ds[i], len(dataset))
            if self.max_samples_per_ds is not None
            else len(dataset)
            for i, dataset in enumerate(self.dataset_list)
        ]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index) -> tuple[dict, Optional[int]]:
        for i, length in enumerate(self.lengths):
            if index < length:
                if self.dataset_indices[i] is not None:
                    # Use random index if available
                    actual_index = self.dataset_indices[i][index]
                else:
                    actual_index = index
                data = self.dataset_list[i][actual_index]  # (time, h, w, n_channels)
                break
            index -= length
        if self.return_ds_idx:
            return data, i
        else:
            return data


def get_all_dt_datasets(
    path: str,
    split_name: str,
    datasets: list,
    num_channels: int,
    min_stride: int = 1,
    max_stride: int = 8,
    use_normalization: bool = True,
    full_trajectory_mode: bool = False,
    nan_to_zero: bool = True,
    split_in_dt: bool = False,
) -> SuperDataset:
    """ """

    all_ds = {}
    for ds_name, traj_length in datasets:
        ds_path = Path(path) / f"{ds_name}/data/{split_name}"
        if ds_path.exists():
            if max_stride == -1:
                max_stride = (
                    traj_length - 2
                )  # leave at least 1 step for input and output

            if split_in_dt:
                strides = range(min_stride, max_stride + 1)
                for stride in strides:
                    dataset = PhysicsDataset(
                        data_dir=Path(path) / f"{ds_name}/data/{split_name}",
                        use_normalization=use_normalization,
                        dt_stride=stride,
                        full_trajectory_mode=full_trajectory_mode,
                        nan_to_zero=nan_to_zero,
                        num_channels=num_channels,
                    )
                    ds_key = f"{ds_name}_dt{stride}"
                    all_ds[ds_key] = dataset
            else:
                dataset = PhysicsDataset(
                    data_dir=Path(path) / f"{ds_name}/data/{split_name}",
                    use_normalization=use_normalization,
                    dt_stride=[min_stride, max_stride],
                    full_trajectory_mode=full_trajectory_mode,
                    nan_to_zero=nan_to_zero,
                    num_channels=num_channels,
                )
                all_ds[ds_name] = dataset

        else:
            print(f"Dataset path {ds_path} does not exist. Skipping.")

    return SuperDataset(all_ds, seed=42)
