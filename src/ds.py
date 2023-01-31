from functools import reduce
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .common import *


class LabelsDataset(Dataset):
    def __init__(self, src: Path) -> None:
        super().__init__()
        self.file_paths = list((src / "images").glob("*.jpg"))

    def __getitem__(self, index) -> torch.Tensor:
        return get_image_and_labels(self.file_paths[index])

    def __len__(self) -> int:
        return len(self.file_paths)


def get_dataloader(src: Path, batch_size: int = 32, num_workers: int = 1):
    src = Path(
        "/home/zuppif/Documents/medium/blazingly-fast-data-pipeline-with-tensordict/data/test/"
    )
    ds = LabelsDataset(src)

    max_num_of_labels = get_max_num_of_labels(src)

    def my_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        images = torch.stack([el[0] for el in batch])
        labels = pad_sequence([el[1] for el in batch])
        return images.contiguous(), labels.contiguous()

    dl = DataLoader(
        ds, batch_size=batch_size, collate_fn=my_collate_fn, num_workers=num_workers
    )

    return dl
