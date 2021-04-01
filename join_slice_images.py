# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:52:34 2021

@author: anton
"""
import os
import shutil
import image_slicer  # !https://github.com/samdobson/image_slicer

from tqdm import tqdm

# =============================================================================
# After whole training process, and then testing LR, I have some slices of
# the LR testing images, so, now I will process these images to join the images
# removing the pad.
# The HR testing images are in: E:\\Hackathon\\TestSet\\600px\\output
# =============================================================================

# %% Test processing
# =============================================================================
# First I move each image_tiles in a folder for each image, so each folder.
# contains the tiles of an image.
# In this step, obtain the image_names to create folders (if it doesn't exist)
# and the move to each folder
# =============================================================================

root = 'E:\\Hackathon\\TestSet\\600px\\output'

filenames = os.listdir(root)
filenames = [os.path.join(root, f) for f in filenames if f.endswith('.png')]

with tqdm(total=len(filenames)) as t:
    for file in filenames:
        folder = file.split('_')[0].split('\\')[3]
        path = root + folder
        try:
            os.mkdir(path)
        except:
            pass
        shutil.move(file, path)
        t.update()


# =============================================================================
# Then I join the tiles for each imagen that are in the folders previouly created.
# It will take like 5 minutes for 100 images sliced in 100 tiles.
# The joined image I rename to candidate_XXXX.png, been XXXX a number like 0001, or 0152
# BE CAREFULL because image_slicer is the module obtained in https://github.com/samdobson/image_slicer
# You can install with: pip install image_slicer
# =============================================================================

filenames = os.listdir(root)
filenames = [os.path.join(root, f) for f in filenames if not f.endswith('.png') and not 'submission' in f]

with tqdm(total=len(filenames)) as t:
    for folder in filenames:
        tiles = image_slicer.open_images_in(folder)
        file = folder.split('\\')[-1]
        image = image_slicer.join(tiles).convert('RGB')
        image.save(folder + f'//candidate_{file}.png')
        t.update()

# =============================================================================
# Then I move the candidate images to a final folder to make submission.
# =============================================================================
foldernames = os.listdir(root)
foldernames = [os.path.join(root, f) for f in foldernames if 'submission' not in f]

with tqdm(total=len(filenames)) as t:
    path = root + '\\final'
    try:
        os.mkdir(path)
    except:
        print("The directory %s already exists" % path)

    for folder in foldernames:
        if not folder.endswith('.png'):
            filenames = os.listdir(folder)
            filenames = [os.path.join(folder, f) for f in filenames if f.startswith('candidate')]
            for files in filenames:
                try:
                    shutil.move(files, path)
                except:
                    print("The file  %s already exists" % files)
                t.update()
                