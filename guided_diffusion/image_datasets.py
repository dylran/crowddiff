import math
import random
import pandas as pd
import cv2
import os

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    normalizer,
    pred_channels,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        normalizer,
        pred_channels,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        normalizer,
        pred_channels,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.normalizer = normalizer
        self.pred_channels = pred_channels

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # get the crowd image
        path = self.local_images[idx]        
        
        image = Image.open(path)
        image = np.array(image.convert('RGB'))
        image = image.astype(np.float32) / 127.5 - 1        

        # get the density map for the image
        path = path.replace('train','train_den').replace('jpg','csv')
        path = path.replace('test','test_den').replace('jpg','csv')

        csv_density = np.asarray(pd.read_csv(path, header=None).values)
        count =  np.sum(csv_density)
        count = np.ceil(count) if count > 1 else count
        csv_density = np.stack(np.split(csv_density, len(self.normalizer), -1))
        csv_density = np.asarray([m/n for m,n in zip(csv_density, self.normalizer)])
        csv_density = csv_density.transpose(1,2,0)

        csv_density = csv_density.clip(0,1)
        csv_density = 2*csv_density - 1
        csv_density = csv_density.astype(np.float32)

        out_dict = {"count": count.astype(np.float32)}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(np.concatenate([csv_density, image], axis=-1), [2, 0, 1]), out_dict


def save_images(image, density, path):
    density = np.repeat(density, 3, axis=-1)
    image = np.concatenate([image, density], axis=1)
    image = 127.5 * (image + 1)

    tag = os.path.basename(path).split('.')[0]
    cv2.imwrite("./results_train/"+tag+'.png', image[:,:,::-1])