# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:52:34 2021

@author: Antonio Guillen-Perez
@twitter: agnprz
@email: antonio_algaida@hotmail.com

"""
import os

from tqdm import tqdm
from myImageSlicer import ImageSlicer

# =============================================================================
# First of all, due to the images are very larges (LR=600x600px; HR=2400x2400px)
# I slice the images into small patches to speed up the I/O in the training
# because in training I need to read a small patch of the original image (32px)
# So I will slice each image in small patches. I will slice each image in 6x6 tiles
# with a pad of 8 pixels to avoid boundaries aberrations.
#
# The original images are in:
#     Training set:
#         LR: E:\\Hackathon\\TrainingSet\\600px
#         HR: E:\\Hackathon\\TrainingSet\\2400px
#     Test set:
#         LR: E:\\Hackathon\\TestSet\\600px
#
# The sliced images will be saved in:
#     Training set:
#         LR: E:\\Hackathon\\TrainingSet\\600px\\croppedoverl
#         HR: E:\\Hackathon\\TrainingSet\\2400px\\croppedoverl
#     Test set:
#         LR: E:\\Hackathon\\TestSet\\600px\\croppedoverl
# =============================================================================
# The images are named like:
#     Training set:
#         LR: E:\\Hackathon\\TrainingSet\\600px\\image_600px_0006.png
#         HR: E:\\Hackathon\\TrainingSet\\2400px\\image_2400px_0006.png
#     Test set:
#         LR: E:\\Hackathon\\TestSet\\600px\\image_600px_1490.png
# =============================================================================
# %% ImageSlicer with padding/overlapping

# =============================================================================
# # Training LR set
# =============================================================================
path = 'E:\\Hackathon\\TrainingSet\\600px'
foldernames = os.listdir(path)
files = [os.path.join(path, f) for f in foldernames if f.endswith('.png')]

original_width = 600
tiles = 6
w = original_width//tiles  # 100x100px
pad = 8
rows = tiles
cols = tiles

with tqdm(total=len(files)) as t:
    for file in files:
        slicer = ImageSlicer(file, size=(w+pad, w+pad), strides=(w, w), PADDING=True)
        transformed_image = slicer.transform()
        img = file.split('_')[2].replace('.png', '')
        slicer.save_images(transformed_image,
                           save_dir='E:\\Hackathon\\TrainingSet\\600px\\croppedoverl',
                           folder_name=img, nrows=rows, ncols=cols)
        t.update()

# =============================================================================
# # Training HR set
# =============================================================================
path = 'E:\\Hackathon\\TrainingSet\\2400px'
foldernames = os.listdir(path)
files = [os.path.join(path, f) for f in foldernames if f.endswith('.png')]

scale = 4
original_width = 600*scale

w = original_width//tiles*scale  # 400x400px
pad = 8*scale
rows = tiles
cols = tiles

with tqdm(total=len(files)) as t:
    for file in files:
        slicer = ImageSlicer(file, size=(w+pad, w+pad), strides=(w, w), PADDING=True)
        transformed_image = slicer.transform()
        img = file.split('_')[2].replace('.png', '')
        slicer.save_images(transformed_image,
                           save_dir='E:\\Hackathon\\TrainingSet\\2400px\\croppedoverl',
                           folder_name=img, nrows=rows, ncols=cols)
        t.update()

# =============================================================================
# # Testing LR set
# # For testing, if you have enough memory (CPU-RAM or GPU-CUDA)
# # you don't have to do this step
# # Because I don't have enough memory, i need to slice the testing images
# =============================================================================
path = 'E:\\Hackathon\\TestSet\\600px'
foldernames = os.listdir(path)
files = [os.path.join(path, f) for f in foldernames if f.endswith('.png')]

tiles = 10
original_width = 600
w = original_width//tiles  # 100x100px
pad = 8
rows = tiles
cols = tiles
with tqdm(total=len(files)) as t:
    for file in files:
        slicer = ImageSlicer(file, size=(w+pad, w+pad), strides=(w, w), PADDING=True)
        transformed_image = slicer.transform()
        img = file.split('_')[2].replace('.png', '')
        slicer.save_images(transformed_image,
                           save_dir='E:\\Hackathon\\TestSet\\600px\\croppedoverl',
                           folder_name=img, nrows=rows, ncols=cols)
        t.update()
