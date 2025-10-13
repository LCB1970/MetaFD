import cv2
from torch.utils.data import Dataset
from PIL import Image
import glob
import torch
import numpy as np
import os
import random
from torchvision import transforms
from torchvision.transforms import functional as TF
import bisect
import warnings
from utils.cfgs import COLORMAP, LABEL_TYPE1, LABEL_TYPE2, LABEL_TYPE3


def image2label(image, dataset_name):
    colormap = COLORMAP[dataset_name]
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1] * 256 + cm[2])] = i
    image = np.int64(image)
    ix = (image[:, :, 0] * 256 + image[:, :, 1] * 256 + image[:, :, 2])
    image2 = cm2lbl[ix]
    return np.int8(image2)


def label2image(prelabel, dataset_name):
    colormap = COLORMAP[dataset_name]
    h, w = prelabel.shape
    prelabel = prelabel.reshape(-1)
    image = np.zeros((h * w, 3), dtype="uint8")
    for ii in range(len(colormap)):
        index = np.where(prelabel == ii)[0]
        image[index, :] = colormap[ii]
    return image.reshape(h, w, 3)


class MyDataset(Dataset):
    def __init__(self, data_root, transform=None, img_format="/*.tif", label_format="/*.tif", filename=False,
                 domain_label=None, add_domain_information=False, dataset_name=None):
        if dataset_name is None:
            dataset_name = os.path.dirname(data_root).split('/')[-2]
        self.data_root = data_root
        img_dir = os.path.join(data_root, 'image')
        label_dir = os.path.join(data_root, 'label')
        self.data_list = glob.glob(img_dir + img_format)
        self.label_list = glob.glob(label_dir + label_format)
        self.dataset_name = dataset_name
        self.transform = transform
        self.filename = filename
        self.domain_label = domain_label
        self.add_domain_information = add_domain_information

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        img = np.asarray(img)
        if img.shape[-1] > 3:
            img = img[:, :, 0:3]
        img = Image.fromarray(img)
        label = Image.open(label)
        img, label = self.apply_transform(img, label, self.transform)
        if self.add_domain_information:
            if self.filename:
                return self.data_list[idx], img, label, self.domain_label
            return img, label, self.domain_label
        else:
            if self.filename:
                return self.data_list[idx], img, label
            return img, label

    def __len__(self):
        return len(self.data_list)

    def apply_transform(self, img, mask, current_transform=None):
        if current_transform is None:
            current_transform = self.transform

        if isinstance(current_transform, (transforms.Compose)):
            for transform in current_transform.transforms:
                img, mask = self.apply_transform(img, mask, transform)

        elif isinstance(current_transform, (transforms.RandomApply)):
            if current_transform.p >= random.random():
                img, mask = self.apply_transform(
                    img, mask, current_transform.transforms
                )

        elif isinstance(current_transform, (transforms.RandomChoice)):
            t = random.choice(current_transform.transforms)
            img, mask = self.apply_transform(img, mask, t)

        elif isinstance(current_transform, (transforms.RandomOrder)):
            order = list(range(len(current_transform.transforms)))
            random.shuffle(order)
            for i in order:
                img, mask = self.apply_transform(
                    img, mask, current_transform.transforms[i]
                )

        elif isinstance(
                current_transform,
                (
                        transforms.CenterCrop,
                        transforms.FiveCrop,
                        transforms.TenCrop,
                        transforms.Grayscale,
                        transforms.Resize,
                ),
        ):
            img = current_transform(img)
            mask = current_transform(mask)

        elif isinstance(current_transform, (transforms.ToTensor)):
            img = current_transform(img)
            if self.dataset_name in LABEL_TYPE1:
                mask = torch.from_numpy(np.asarray(mask) // 255)
            elif self.dataset_name in LABEL_TYPE2:
                mask = torch.from_numpy(np.asarray(mask))
            elif self.dataset_name in LABEL_TYPE3:
                mask = np.asarray(mask)
                mask = image2label(mask, self.dataset_name)
                mask = torch.from_numpy(mask)



        elif isinstance(
                current_transform, (transforms.Normalize, transforms.Lambda, transforms.Pad)
        ):
            img = current_transform(img)
            # mask = current_transform(mask)  # apply on input only

        elif isinstance(current_transform, (transforms.ColorJitter)):
            img = current_transform(img)

        elif isinstance(current_transform, (transforms.RandomAffine)):
            ret = current_transform.get_params(
                current_transform.degrees,
                current_transform.translate,
                current_transform.scale,
                current_transform.shear,
                img.size,
            )
            img = TF.affine(
                img,
                *ret,
                interpolation=current_transform.interpolation,
                fill=current_transform.fill,
            )
            mask = TF.affine(
                mask,
                *ret,
                interpolation=current_transform.interpolation,
                fill=current_transform.fill,
            )

        elif isinstance(current_transform, (transforms.RandomCrop)):
            i, j, h, w = current_transform.get_params(img, current_transform.size)
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        elif isinstance(current_transform, (transforms.RandomHorizontalFlip)):
            if random.random() < current_transform.p:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        elif isinstance(current_transform, (transforms.RandomVerticalFlip)):
            if random.random() < current_transform.p:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

        elif isinstance(current_transform, (transforms.RandomPerspective)):
            if random.random() < current_transform.p:
                width, height = img.size
                startpoints, endpoints = current_transform.get_params(
                    width, height, current_transform.distortion_scale
                )
                img = TF.perspective(
                    img, startpoints, endpoints, current_transform.interpolation
                )
                mask = TF.perspective(
                    mask, startpoints, endpoints, current_transform.interpolation
                )

        elif isinstance(current_transform, (transforms.RandomResizedCrop)):
            ret = current_transform.get_params(
                img, current_transform.scale, current_transform.ratio
            )
            img = TF.resized_crop(
                img, *ret, current_transform.size, current_transform.interpolation
            )
            mask = TF.resized_crop(
                mask, *ret, current_transform.size, current_transform.interpolation
            )

        elif isinstance(current_transform, (transforms.RandomRotation)):
            angle = current_transform.get_params(current_transform.degrees)

            img = TF.rotate(
                img,
                angle,
                current_transform.interpolation,
                current_transform.expand,
                current_transform.center,
            )
            mask = TF.rotate(
                mask,
                angle,
                current_transform.interpolation,
                current_transform.expand,
                current_transform.center,
            )

        else:
            raise NotImplementedError(
                f'Transform "{current_transform}" not implemented yet'
            )
        return img, mask

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



