from tensordict import MemmapTensor, TensorDict
from tensordict.prototype import tensorclass
from tqdm import tqdm
from pathlib import Path
from torchvision.io import read_image
import torch


def get_labels(file_path: Path) -> torch.Tensor:
    label_file_path = file_path.parent.parent / "labels" / f"{file_path.stem}.txt"
    with label_file_path.open("r") as f:
        # reading all labels in one go
        # ['label x y h w', ....]
        labels_raw = f.read().split("\n")
        labels_raw = [row.split(" ") for row in labels_raw]
        labels = torch.tensor([[float(el) for el in row] for row in labels_raw])
    return labels


def get_image_and_labels(file_path: Path) -> tuple[torch.Tensor, torch.IntTensor]:
    labels = get_labels(file_path)
    image = read_image(str(file_path))
    return image, labels


def get_file_paths(src: Path, fmt: str = "jpg") -> list[Path]:
    return src.glob(f"**/*.{fmt}")


@tensorclass
class ObjectDetectionData:
    images: torch.Tensor
    labels: torch.Tensor
    # bboxes: torch.Tensor

    @classmethod
    def from_dataset(cls, src: Path):
        file_paths = list(get_file_paths(src))
        pipe = map(get_image_and_labels, file_paths)
        img, _ = next(pipe)

        data: TensorDict = cls(
            images=MemmapTensor(
                len(file_paths),
                *img.shape,
                dtype=torch.uint8,
            ),
            labels=MemmapTensor(len(file_paths), dtype=torch.int64),
            # bboxes=MemmapTensor(len(file_paths), dtype=torch.float32),
            batch_size=[len(file_paths)],
        )
        data = data.memmap_()

        # dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        i = 0
        pbar = tqdm(total=len(file_paths))
        for image, labels in pipe:
            # _batch = image.shape[0]
            print(image.unsqueeze(0).shape)
            print(labels[:,0].unsqueeze(0).shape)
            print(labels[:,1:].unsqueeze(0).shape)
            _batch = 1
            pbar.update(_batch)
            data[i : i + _batch] = cls(
                images=image.unsqueeze(0),
                labels=labels[:,0].unsqueeze(0),
                # bboxes=labels[:,1:].unsqueeze(0),
                batch_size=[_batch],
            )
            i += _batch

        return data
