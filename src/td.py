from functools import reduce
from pathlib import Path
from time import perf_counter

import torch
from tensordict import MemmapTensor, TensorDict
from tensordict.prototype import tensorclass
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import *


@tensorclass
class ObjectDetectionData:
    images: torch.Tensor
    labels: torch.Tensor
    labels_offsets: torch.IntTensor

    @classmethod
    def from_dataset(cls, src: Path):
        file_paths = list(get_file_paths(src))
        pipe = map(get_image_and_labels, file_paths)
        img, _ = next(pipe)
        num_images = len(file_paths)
        max_num_of_labels = get_max_num_of_labels(src)
        # N = num of images, M = bboxes per image -> NxMx...
        data: TensorDict = cls(
            images=MemmapTensor(
                num_images,
                *img.shape,
                dtype=torch.uint8,
            ),
            labels=MemmapTensor(num_images, max_num_of_labels, 5, dtype=torch.float32),
            labels_offsets=MemmapTensor(num_images, dtype=torch.long),
            # bboxes=MemmapTensor(num_images, dtype=torch.float32),
            batch_size=[num_images],
        )
        data = data.memmap_()

        # dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        i = 0
        pbar = tqdm(total=len(file_paths))
        for image, labels in pipe:
            num_labels = labels.shape[0]
            padded_labels = torch.empty((max_num_of_labels, 5))
            padded_labels[:num_labels] = labels
            _batch = 1
            pbar.update(_batch)
            data[i : i + _batch] = cls(
                images=image.unsqueeze(0),
                labels=padded_labels.unsqueeze(0),
                labels_offsets=torch.tensor([num_labels], dtype=torch.long).unsqueeze(
                    0
                ),
                batch_size=[_batch],
            )
            i += _batch

        return data


def get_dataloader(src: Path, batch_size: int = 32, num_workers: int = 1):
    td = ObjectDetectionData.from_dataset(src)
    td.apply(lambda x: x.contiguous())

    def my_collate_fn(batch):
        batch.cuda()
        # batch = batch.apply(lambda x: x.contiguous())

        return batch

    dl = DataLoader(td, batch_size=batch_size, collate_fn=my_collate_fn)

    return dl
