from enum import Enum
from pathlib import Path

from scalene import scalene_profiler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ds import get_dataloader as ds_get_dataloader
from src.td import get_dataloader as td_get_dataloader

from time import perf_counter

class DataType(str, Enum):
    tensordict = "td"
    dataset = "ds"


import typer


def main(
    root: Path,
    batch_size: int = 32,
    num_workers: int = 1,
    data_type: DataType = DataType.tensordict,
    epoches: int = 10,
):
    # scalene_profiler.stop()
    data_type_dl_funcs = {
        DataType.tensordict: td_get_dataloader,
        DataType.dataset: ds_get_dataloader,
    }

    scalene_profiler.start()

    dl: DataLoader = data_type_dl_funcs[data_type](root, batch_size, num_workers)
    # scalene_profiler.stop()

    for _ in range(epoches):
        start = perf_counter()

        for data in tqdm(dl):
            continue

        print(f"took {(perf_counter() - start) * 1000:.4f}ms")
    scalene_profiler.stop()


if __name__ == "__main__":
    typer.run(main)
