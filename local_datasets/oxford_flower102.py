# local_datasets/oxford_flower102.py

from typing import Callable, Optional

from torchvision import datasets


# Labels in torchvision Flowers102 are usually 1..102 internally in _labels,
# but torchvision returns labels as 0..101 in __getitem__ in modern versions.
# These class names are ordered according to Oxford Flowers 102 classes.
FLOWERS102_CLASS_NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


def _get_flowers_targets(dataset) -> list[int]:
    if hasattr(dataset, "_labels"):
        labels = [int(x) for x in dataset._labels]

        # Defensive normalization:
        # if labels are 1..102, convert to 0..101.
        if min(labels) == 1 and max(labels) == 102:
            labels = [x - 1 for x in labels]

        return labels

    if hasattr(dataset, "labels"):
        labels = [int(x) for x in dataset.labels]
        if min(labels) == 1 and max(labels) == 102:
            labels = [x - 1 for x in labels]
        return labels

    if hasattr(dataset, "targets"):
        labels = [int(x) for x in dataset.targets]
        if min(labels) == 1 and max(labels) == 102:
            labels = [x - 1 for x in labels]
        return labels

    raise ValueError("Cannot find labels for Flowers102 dataset.")


class OxfordFlowers102:
    def __init__(
        self,
        root: str = "./data",
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = root
        self.download = download
        self.num_classes = 102
        self.class_names = FLOWERS102_CLASS_NAMES

        self.train_dataset = datasets.Flowers102(
            root=root,
            split="train",
            transform=transform,
            download=download,
        )

        self.val_dataset = datasets.Flowers102(
            root=root,
            split="val",
            transform=transform,
            download=download,
        )

        self.test_dataset = datasets.Flowers102(
            root=root,
            split="test",
            transform=transform,
            download=download,
        )

        self._patch_metadata()

    def _patch_metadata(self):
        self.train_dataset.targets = _get_flowers_targets(self.train_dataset)
        self.val_dataset.targets = _get_flowers_targets(self.val_dataset)
        self.test_dataset.targets = _get_flowers_targets(self.test_dataset)

        self.train_dataset.class_names = self.class_names
        self.val_dataset.class_names = self.class_names
        self.test_dataset.class_names = self.class_names

    def set_transforms(
        self,
        train_transform=None,
        val_transform=None,
        test_transform=None,
    ):
        if train_transform is not None:
            self.train_dataset.transform = train_transform

        if val_transform is not None:
            self.val_dataset.transform = val_transform

        if test_transform is not None:
            self.test_dataset.transform = test_transform
