import functools
import logging
import random
from collections.abc import Callable, Generator, Sequence
from typing import Any, Literal
from operator import call
import torch
import SimpleITK as sitk
import numpy as np
from myai.torch_tools import pad_to_shape

def binary_threshold(t:torch.Tensor, threshold:float):
    return torch.where(t > threshold, 1, 0)

class MRISlicer:
    def __init__(
        self,
        data: torch.Tensor | np.ndarray | Sequence[sitk.Image],
        seg: torch.Tensor,
        num_classes: int | None = None,
        around: int = 1,
        any_prob: float = 0.1,
        warn_empty=True,
    ):
        """MRI slicer for 2D slice-wise training.

        :param data: 4-D tensor or ndarray of (C, D, H, W) shape, where C is channels (e.g. different modalities).
        :param seg: Segmentation of the same shape as `data[0]`, either one-hot encoded 4-D tensor of (D, C, H, W) shape, where D is the number of classes, or 3D tensor of (D, H, W) shape.
        :param num_classes: Number of classes, if None, inferred from `D` dimension of `seg`. But if `seg` is not one-hot encoded, this must be provided. defaults to None
        :param around: How many neigbouring slices to take around a 2D slice, defaults to 1
        :param any_prob: Probability to pick a random slice. Otherwise, a random slice is picked from all slices that contain any segmentation. defaults to 0.1.
        :param warn_empty: Whether to warn if `seg` is empty. defaults to True
        """
        if isinstance(data, np.ndarray): data = torch.from_numpy(data)
        elif not isinstance(data, torch.Tensor):
            data = torch.from_numpy(np.array([sitk.GetArrayFromImage(i) for i in data]))


        if data.ndim != 4: raise ValueError(f"`tensor` is {data.shape}")
        if seg.ndim not in (3, 4): raise ValueError(f"`seg` is {seg.shape}")
        if seg.ndim == 4:
            if num_classes is None: num_classes = seg.size(0)
            elif num_classes < seg.size(0): raise ValueError(f'{seg.shape = } (first dimension should be same as num_classes), but {num_classes = }')
            seg = seg.argmax(0)
        elif num_classes is None: raise ValueError('`num_classes` needs to be specified when `seg` is not one-hot encoded.')

        self.data: torch.Tensor = data
        """C, D, H, W"""
        self.seg: torch.Tensor = seg
        """D, H, W, not one hot encoded"""
        self.num_classes: int = num_classes

        if self.data.shape[1:] != self.seg.shape:
            raise ValueError(f"Shapes don't match: image is {self.data.shape}, seg is {self.seg.shape}")

        self.x,self.y,self.z = [],[],[]

        # here we use binary threshold to pick any segmentation that is not 0 (background)
        # and save indexes of all slices that have segmentation on them
        # save top
        seg = seg.float()
        for i, sl in enumerate(binary_threshold(seg, 0)):
            if sl.sum() > 0: self.x.append(i)

        # save front
        for i, sl in enumerate(binary_threshold(seg.swapaxes(0,1), 0)):
            if sl.sum() > 0: self.y.append(i)

        # save side
        for i, sl in enumerate(binary_threshold(seg.swapaxes(0,2), 0)):
            if sl.sum() > 0: self.z.append(i)

        if len(self.x) == 0:
            if warn_empty: logging.warning('Segmentation is empty, setting probability to 0.')
            self.any_prob = 0

        self.shape = self.data.shape
        self.around = around
        self.any_prob = any_prob

    def set_settings(self, around: int | None = None, any_prob: float | None = None):
        if around is not None: self.around = around
        if len(self.x) > 0 and any_prob is not None: self.any_prob = any_prob

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll `any_prob`, pick a random axis, and in that axis pick either a random slice that contains segmentation, or any random slice. Returns that slice plus neigbouring slices if `around` > 0.

        Returns a (mri, segmentation) slice.

        MRI is a [C, H, W] tensor, where C = num_channels * (1 + 2*around).

        For example, if you have 2 channels: t1 and t2, and around = 1, the 6 resulting channels will be:

        ```
        [t1[coord - 1], t1[coord], t1[coord + 1], t2[coord - 1], t2[coord], t2[coord + 1]
        ```

        Segmentation is a [H,W] tensor (seg[coord]), not one-hot encoded.
        """
        # pick a dimension
        dim: Literal[0,1,2] = random.choice([0,1,2])

        # get length
        if dim == 0: length = self.shape[1]
        elif dim == 1: length = self.shape[2]
        else: length = self.shape[3]

        # pick a coord
        # from segmentation
        if random.random() > self.any_prob:
            if dim == 0: coord = random.choice(self.x)
            elif dim == 1: coord = random.choice(self.y)
            else: coord = random.choice(self.z)

        else:
            coord = random.randrange(self.around, length - self.around)

        return self.get_slice(dim, coord)

    def get_slice(self, dim: Literal[0,1,2], coord: int, randflip = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a slice along given `dim` and `coord`, plus neigbouring slices if `around` > 0.

        Returns a (mri, segmentation) slice.

        MRI is a [C, H, W] tensor, where C = num_channels * (1 + 2*around).

        For example, if you have 2 channels: t1 and t2, and around = 1, the 6 resulting channels will be:

        ```
        [t1[coord - 1], t1[coord], t1[coord + 1], t2[coord - 1], t2[coord], t2[coord + 1]
        ```

        Segmentation is a [H,W] tensor (seg[coord]), not one-hot encoded.
        """
        # get a tensor
        if dim == 0:
            tensor = self.data
            seg = self.seg
            length = self.shape[1]
        elif dim == 1:
            tensor = self.data.swapaxes(1, 2)
            seg = self.seg.swapaxes(0,1)
            length = self.shape[2]
        else:
            tensor = self.data.swapaxes(1, 3)
            seg = self.seg.swapaxes(0,2)
            length = self.shape[3]

        # check if coord outside of bounds
        if coord < self.around: coord = self.around
        elif coord + self.around >= length: coord = length - self.around - 1


        # get slice
        if self.around == 0: return tensor[:, coord], seg[coord]

        # or get slices around (and flip slice spatial dimension with 0.5 p)
        if randflip:
            if random.random() > 0.5: return tensor[:, coord - self.around : coord + self.around + 1].flatten(0,1), seg[coord]
            return tensor[:, coord - self.around : coord + self.around + 1].flip((1,)).flatten(0,1), seg[coord]
        return tensor[:, coord - self.around : coord + self.around + 1].flatten(0,1), seg[coord]

    def get_random_slice(self):
        """Get a random slice, ignores `any_prob`."""
        # pick a dimension
        dim: Literal[0,1,2] = random.choice([0,1,2])

        # get length
        if dim == 0: length = self.shape[1]
        elif dim == 1: length = self.shape[2]
        else: length = self.shape[3]

        coord = random.randrange(0 + self.around, length - self.around)
        return self.get_slice(dim, coord)

    def yield_all_seg_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices that have segmentation as callables."""
        # pick a dimension
        for dim in (0, 1, 2):

            if dim == 0: coord_list = self.x
            elif dim == 1: coord_list = self.y
            else: coord_list = self.z

            for coord in coord_list:

                yield functools.partial(self.get_slice, dim, coord)

    def get_all_seg_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Get a list of all slices that have segmentation as callables."""
        return list(self.yield_all_seg_slice_callables())

    def get_all_seg_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get a list of all slices that have segmentation."""
        return [i() for i in self.get_all_seg_slice_callables()]


    def yield_all_dim_slice_callables(self, dim:Literal[0,1,2], randflip=False) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices along a specified dimension as callables."""
        if dim == 0: length = self.shape[1]
        elif dim == 1: length = self.shape[2]
        else: length = self.shape[3]

        for coord in range(self.around, length - self.around):
            yield functools.partial(self.get_slice, dim, coord, randflip)

    def get_all_dim_slice_callables(self, dim:Literal[0,1,2], randflip=False) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Get a list of all slices along a specified dimension as callables."""
        return list(self.yield_all_dim_slice_callables(dim, randflip))

    def get_all_dim_slices(self, dim:Literal[0,1,2], randflip=False) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get a list of all slices along a specified dimension."""
        return [i() for i in self.get_all_dim_slice_callables(dim, randflip)]

    def yield_all_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices, including ones that have no segmentation, as callables."""
        # pick a dimension
        for dim in (0, 1, 2): yield from self.yield_all_dim_slice_callables(dim)

    def get_all_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Get a list of all slices, including ones that have no segmentation, as callables."""
        return list(self.yield_all_slice_callables())

    def get_all_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get a list of all slices, including ones that have no segmentation."""
        return [i() for i in self.get_all_slice_callables()]


    def yield_all_empty_slice_callables(self) -> Generator[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Yield all slices that have no segmentation as callables."""
        # pick a dimension
        for dim in (0, 1, 2):

            # get length
            if dim == 0:
                coord_list = self.x
                length = self.shape[1]
            elif dim == 1:
                coord_list = self.y
                length = self.shape[2]
            else:
                coord_list = self.z
                length = self.shape[3]
            for coord in range(self.around, length - self.around):
                if coord not in coord_list: yield functools.partial(self.get_slice, dim, coord)

    def get_all_empty_slice_callables(self) -> list[Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
        """Get a list of all slices that have no segmentation as callables."""
        return list(self.yield_all_empty_slice_callables())

    def get_all_empry_slices(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get a list of all slices that have no segmentation."""
        return [i() for i in self.get_all_empty_slice_callables()]

    def get_non_empty_count(self):
        """Returns the total amount of slices in along all 3 dimensions that have segmentation on them"""
        return len(self.x) + len(self.y) + len(self.z)

    def get_anyp_random_slice_callables(self):
        """Returns a list of slice callables, including all slices that have segmentation, plus a few callables that return a random slice. In total `self.any_prob` * 100 % callables will be random."""
        seg_prob = 1 - self.any_prob
        any_to_seg_ratio = self.any_prob / seg_prob
        return [self.get_random_slice for i in range(int(self.get_non_empty_count() * any_to_seg_ratio))]

# def loader(x:MRISlicer | Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
#     """Loader for MRISlicer slice callables. Simply calls them."""
#     return x()

loader = call
"""calls mrislicer slice callables"""

def randcrop2dt(x: tuple[torch.Tensor, torch.Tensor], size = (96,96)):
    """Random crop transform for MRISlicer slices, randomly crops the slice to `size`.
    But you can use it any other images, as long as they are in the right format.

    x needs to be tuple of `(image, seg)`.
    `image` needs to be `(C, H, W)`, `seg` needs to be `(H, W)` or `(C, H, W)`.
    """
    image, seg = x

    # check ndims
    if image.ndim != 3: raise ValueError(f'randcrop: {image.shape = } and should have 3 dims')
    if seg.ndim not in (2, 3): raise ValueError(f'randcrop: {seg.shape = } and should have 2 or 3 dims')

    # check that shapes match
    if image.shape[-2:] != seg.shape[-2:]:
        raise ValueError(f'randcrop: {image.shape = } and {seg.shape = } do not match')

    # check if x shape matches size
    if image.shape[1] == size[0] and image.shape[2] == size[1]:
        return x

    # check if x too small
    if image.shape[1] <= size[0] or image.shape[2] <= size[1]:
        #raise ValueError(f'image is {image.shape = }, which is too small to crop to given {size = }')
        image = pad_to_shape(image, [max(image.shape[1], size[0]), max(image.shape[2], size[1])], mode = 'min')
        seg = pad_to_shape(seg, [max(image.shape[1], size[0]), max(image.shape[2], size[1])], value = 0)
    #     return (crop_to_shape(pad_to_shape(x[0], (x[0].shape[0], *size), where='center', mode = 'min'), (x[0].shape[0], *size)).to(torch.float32),
    #             crop_to_shape(pad_to_shape(x[1], size, where='center', value = 0), size).to(torch.float32))

    if image.shape[1] == size[0]: startx = 0
    else: startx = random.randint(0, (image.shape[1] - size[0]) - 1)

    if image.shape[2] == size[1]: starty = 0
    else: starty = random.randint(0, (image.shape[2] - size[1]) - 1)

    return (image[:, startx:startx+size[0], starty:starty+size[1]],
            seg[..., startx:startx+size[0], starty:starty+size[1]], )

def shuffle_channels(x:torch.Tensor):
    """Shuffle first axis in a C* tensor"""
    return x[torch.randperm(x.shape[0])]

def shuffle_channel_groups(x:torch.Tensor, channels_per: int):
    """Shuffle first axis in a C* tensor in groups of `channels_per`.
    (set channels per to 2*around + 1)"""
    num_groups = int(x.shape[0] / channels_per)
    perm = torch.randperm(num_groups, dtype=torch.int32)
    img= x.reshape(num_groups, channels_per, *x.shape[1:])[perm].flatten(0, 1)
    return img

class ShuffleChannelGroups:
    def __init__(self, channels_per: int, p: float):
        self.channels_per = channels_per
        self.p = p
    def __call__(self, x: torch.Tensor):
        return shuffle_channel_groups(x, self.channels_per) if random.random() < self.p else x


def groupwise_apply(x:torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], channels_per = 3):
    """Apply `fn` to each `channels_per` group of channels in a C* tensor."""
    n_channels = x.shape[0]
    assert n_channels % channels_per == 0, f'x.shape[0] must be divisible by channels_per, but {x.shape[0]} is not divisible by {channels_per}'
    num_groups = int(n_channels / channels_per)
    groups = x.reshape(num_groups, channels_per, *x.shape[1:]).unbind(0)
    groups = [fn(i) for i in groups]
    return torch.cat(groups, 0)

def groupwise_apply_batched(x:torch.Tensor, fn: Callable[[torch.Tensor], torch.Tensor], channels_per = 3):
    """Apply `fn` to each `channels_per` group of channels in a C* tensor, input tensor to fn is (B, H, W)"""
    batch_size, n_channels = x.shape[0], x.shape[1]
    assert n_channels % channels_per == 0, f'x.shape[0] must be divisible by channels_per, but {x.shape[0]} is not divisible by {channels_per}'
    num_groups = int(n_channels / channels_per)
    groups = x.reshape(batch_size, num_groups, channels_per, *x.shape[2:]).unbind(1)
    groups = [fn(i) for i in groups]
    return torch.cat(groups, 1)

class GroupwiseApply:
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], channels_per: int):
        self.fn = fn
        self.channels_per = channels_per
    def __call__(self, x: torch.Tensor): return groupwise_apply(x, self.fn, self.channels_per)