"""Upstream's ImageNetSR pipeline, pointed at a file list instead of ImageNet.

The resizing/cropping strategy is not reimplemented here. `ImageNetSR.__getitem__`
-- random scale crop of `min_side * U(min_crop_f, max_crop_f)`, RandomCrop at
that size, then SmallestMaxSize(size, cv2.INTER_AREA), then `x/127.5 - 1` -- is
inherited verbatim. The only thing overridden is `get_base()`, because upstream's
is welded to ImageNet:

    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)
        dset = ImageNetTrain(process_images=False,)
        return Subset(dset, indices)

It reads a pickled ImageNet index and instantiates ImageNetTrain, which
auto-downloads ImageNet; there is no way to aim it at a directory. `get_base`
only has to return something indexable yielding dicts with a "file_path_" key,
which is all this supplies.
"""

from torch.utils.data import Dataset

from ldm.data.imagenet import ImageNetSR


class _FileListBase(Dataset):
    """Minimal stand-in for upstream's Subset(ImageNetTrain, indices).

    Returns a *fresh* dict each call, matching upstream's ImagePaths: consumers
    mutate the example in place (adding "image" / "LR_image"), so handing back a
    shared dict would accumulate arrays across epochs.
    """

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"file_path_": self.paths[i]}


class _FileListSR(ImageNetSR):
    def __init__(self, images_list_file, **kwargs):
        # ImageNetSR.__init__ calls self.get_base() as its first statement, so
        # this has to be set before delegating up.
        self.images_list_file = images_list_file
        super().__init__(**kwargs)

    def get_base(self):
        with open(self.images_list_file, "r") as f:
            paths = [p for p in f.read().splitlines() if p]
        return _FileListBase(paths)


class FileListSRTrain(_FileListSR):
    """Drop-in for ldm.data.imagenet.ImageNetSRTrain over an explicit file list."""


class FileListSRValidation(_FileListSR):
    """Drop-in for ldm.data.imagenet.ImageNetSRValidation over an explicit file list."""
