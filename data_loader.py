# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:52:34 2021

@author: Antonio Guillen-Perez
@twitter: agnprz
@email: antonio_algaida@hotmail.com
"""

import random
import os
import csv

import numpy as np

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from skimage.feature import canny


def random_crop(HQ, LQ, patch_size, scale=4):
    h, w = np.shape(LQ)[:2]
    x = random.randrange(0, w-patch_size+1)
    y = random.randrange(0, h-patch_size+1)

    crop_HQ = HQ.crop((x*scale, y*scale,
                       x*scale+patch_size*scale, y*scale+patch_size*scale))
    crop_LQ = LQ.crop((x, y,
                       x+patch_size, y+patch_size))

    return crop_HQ.copy(), crop_LQ.copy()


def flip_and_rotate_and_edges(HQ, LQ, HQ_edge, LQ_edge):
    hfli = random.random() < 0.5
    vfli = random.random() < 0.5
    rot90 = random.random() < 0.5

    if hfli:
        HQ = F.hflip(HQ)
        LQ = F.hflip(LQ)
        HQ_edge = F.hflip(HQ_edge)
        LQ_edge = F.hflip(LQ_edge)

    if vfli:
        HQ = F.vflip(HQ)
        LQ = F.vflip(LQ)
        HQ_edge = F.vflip(HQ_edge)
        LQ_edge = F.vflip(LQ_edge)

    if rot90:
        HQ = F.rotate(HQ, 90)
        LQ = F.rotate(LQ, 90)
        HQ_edge = F.rotate(HQ_edge, 90, fill=(0,))
        LQ_edge = F.rotate(LQ_edge, 90, fill=(0,))

    return HQ, LQ, HQ_edge, LQ_edge


def normalize(tensors):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    """ Denormalizes image tensors using mean and std """
    res = tensors.detach().clone()
    for c in range(3):
        res[:, c] = tensors[:, c].sub(std[c]).div(mean[c])
    return res


def load_edge(img):
    return canny(np.array(img.convert('L')), sigma=2).astype(np.float32)


def random_crop_edges(hr_img, lr_img, hr_edge, lr_edge, patch_size, scale=4):
    try:
        h, w = np.shape(lr_img)[:2]
        x = random.randrange(0, w-patch_size+1)
        y = random.randrange(0, h-patch_size+1)

        crop_HQ = hr_img.crop((x*scale, y*scale, x*scale+patch_size *
                               scale, y*scale+patch_size*scale))
        crop_LQ = lr_img.crop((x, y, x+patch_size, y+patch_size))

        crop_HQ_edge = hr_edge.crop((x*scale, y*scale, x*scale+patch_size *
                                     scale, y*scale+patch_size*scale))
        crop_LQ_edge = lr_edge.crop((x, y, x+patch_size, y+patch_size))
        return crop_HQ, crop_LQ, crop_HQ_edge, crop_LQ_edge

    except Exception as e:
        print(e)
        return hr_img, lr_img, hr_edge, lr_edge


class TrainDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    # Note that the first directory is train, the second directory is label
    
    def __init__(self, LR_dir, HR_dir, batch_size=8, max_patch=45):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            HR_dir: (string) directory containing the high resolution dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.HR_filenames = os.listdir(HR_dir)  # label
        self.HR_filenames = [os.path.join(HR_dir, f) for f in self.HR_filenames if f.endswith('.png')]  # label

        # Here copy the filenames of labels, labels == blur_filenames
        self.LR_filenames = os.listdir(LR_dir)  # train
        self.LR_filenames = [os.path.join(LR_dir, f) for f in self.LR_filenames if f.endswith('.png')]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.batch_size = batch_size
        self.n_it = 0
        self.patch_size = 24
        self.max_patch = 40
        self.min_patch = 16

    def __len__(self):
        # return size of dataset
        return len(self.HR_filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed blur image
            label: (Tensor) transformed original image
        """
        
        # Random patch_size
        # Due to that each image in each batch for training needs to be the same
        # size, I change the patch_size each batch_size images.
        
        if self.n_it == 0:
            self.patch_size = random.randrange(self.min_patch, int(self.max_patch-self.patch_size/2))
        self.n_it = (self.n_it+1) % self.batch_size

        hr_img = Image.open(self.HR_filenames[idx]).convert('RGB')
        lr_img = Image.open(self.LR_filenames[idx]).convert('RGB')
        
        # At this point, the images are between [0,1]
        
        # load edge
        hr_edge = load_edge(hr_img)
        lr_edge = load_edge(lr_img)
        
        # Data augmentation
        # Random flip and rotation
        hr_img, lr_img, hr_edge, lr_edge = flip_and_rotate_and_edges(hr_img, lr_img, Image.fromarray(hr_edge), Image.fromarray(lr_edge))

        # Brightness
        if random.random() > 0.5:
            r = random.uniform(0.8, 1.2)
            hr_img = F.adjust_brightness(hr_img, r)
            lr_img = F.adjust_brightness(lr_img, r)
        
        # Contrast
        if random.random() > 0.5:
            r = random.uniform(0.8, 1.2)
            hr_img = F.adjust_contrast(hr_img, r)
            lr_img = F.adjust_contrast(lr_img, r)

        # Saturation
        if random.random() > 0.5:
            r = random.uniform(0.5, 1.5)
            hr_img = F.adjust_saturation(hr_img, r)
            lr_img = F.adjust_saturation(lr_img, r)

        # HUE
        if random.random() > 0.5:
            r = random.random()-0.5
            hr_img = F.adjust_hue(hr_img, r)
            lr_img = F.adjust_hue(lr_img, r)

        # Random cropping with size self.patch_siz
        hr_img, lr_img, hr_edge, lr_edge = random_crop_edges(hr_img, lr_img,
                                                             hr_edge, lr_edge,
                                                             self.patch_size)
        
        # Transform and scale between [-1, +1]
        return self.transform(lr_img)*2-1, self.transform(hr_img)*2-1, self.transform(lr_edge), self.transform(hr_edge)


class TestDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """

    # Note that the first directory is train, the second directory is label
    def __init__(self, LR_data_dir):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.
        Args:
            transform: (torchvision.transforms) transformation to apply on image
        """

        # Here copy the filenames of labels, labels == blur_filenames
        self.LR_filenames = os.listdir(LR_data_dir)
        self.LR_filenames = [os.path.join(LR_data_dir, f) for f in self.LR_filenames if f.endswith('.png')]

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        # return size of dataset
        return len(self.LR_filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed blur image
            label: (Tensor) transformed original image
        """

        # Load test image
        train_image = Image.open(self.blur_filenames[idx]).convert('RGB')
        
        # Load LR edges
        lr_edge = load_edge(train_image)
        
        # Transform and scale between [-1, +1]
        return self.transform(train_image)*2-1, self.transform(lr_edge), self.blur_filenames[idx]


def fetch_dataloader(splits='train', batch_size=8, max_patch=40, group=0, nworks=1):
    # Here add a new parameter "blur_data_dir", which is train
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        splits: (list) has one or more of 'train', 'test' depending on which data is required
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """

    dataloaders = {}
    for split in [splits]:
        if split == 'train':
            data_dir = 'TrainingSet'
            path_blur = data_dir+'/600px/croppedoverl'
            path = data_dir+'/2400px/croppedoverl'
            dl = DataLoader(TrainDataset(path_blur, path, batch_size=batch_size,
                                         max_patch=max_patch), batch_size=batch_size,
                            shuffle=True, num_workers=nworks)

        if split == 'test':
            data_dir = 'TestSet'
            path_blur = data_dir+'/600px/croppedoverl'
            dl = DataLoader(TestDataset(path_blur),
                            batch_size=1, shuffle=False)

        dataloaders[split] = dl

    return dataloaders
