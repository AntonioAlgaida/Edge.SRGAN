# Edge.SRGAN
This repository was created in order to participate in the Hackathon organized by @SpainAI, in the computer vision challenge. The objective of this challenge was the generation of high resolution images, i.e. Single Image Super Resolution (SISR). For this, I decided to implement a solution that unifies the advantages offered by SRGAN (see https://arxiv.org/abs/1609.04802) together with those offered by the edge prediction (Edge Informed SISR) introduced in https://arxiv.org/abs/1909.05305.

## Analyzing the challenge:
In this challenge we were asked to train a system that learns to generate high resolution images from low quality images. For this, we provided a training dataset where low quality images existed, as well as the corresponding high resolution images for each of these images.
In addition, another set of low resolution test data was provided and used to evaluate the proposed solutions.
What you were asked is that for the low quality test image set, generate the high quality images.
This challenge was posed by looking for a practical application of Generative Adversarial Neural Networks (GANs) algorithms.

![SISR](https://beyondminds.ai/wp-content/uploads/2020/07/1_bfLS2BU_d7HMkzwF8aUbDg.png)

[SISR with GANs - https://beyondminds.ai/blog/an-introduction-to-super-resolution-using-deep-learning/]

The metric used to evaluate the solutions was the Structural similarity index (SSIM, see https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e.)

## Analyzing the dataset:
As a set of supplied data, we have two folders, one for training and one for testing.
Inside each folder there is a folder for the low resolution images (600x600 px) and another one for the high resolution images (2400x2400 px, only in the training dataset).

The images have the name: image_[_resolution_]_[_id_].png
- where [_resolution_]: "600px" or "2400px"
- and [_id_]: an integer between "0000" and "2105" that identifies each image.

In summary, we have:
- The original images are in:
> Training set:
> - LR: TrainingSet\\600px
> - HR: TrainingSet\\2400px

>Test set:
> - LR: TestSet\\600px

- The images are named like:
> Training set:
> - LR: TrainingSet\\600px\\image_600px_0006.png
> - HR: TrainingSet\\2400px\\image_2400px_0006.png

> Test set:
> - LR: TestSet\\600px\\image_600px_1490.png

## Preprocessing the images:
First, because the images are very large (LR=600x600px; HR=2400x2400px). I cut the images into small patches to speed up the I/O in training because in training I need to read a small patch of the original image (32px). So I'm going to cut each image into small patches. I will cut each image into 6x6 tiles with an 8 pixels pad to avoid boundary aberrations.<br>
To do this, you must run 

> slice_images_w_overlapping.py

Read carefully the comments in this file to obtain the tiles of each image
The sliced images will be saved in:

> Training set:
> - LR: TrainingSet\\600px\\croppedoverl
> - HR: TrainingSet\\2400px\\croppedoverl

> Test set:
> - LR: TestSet\\600px\\croppedoverl

## Edge generator:
To obtain a edge generator I use the code in Edge Informed SISR code (https://github.com/knazeri/edge-informed-sisr)<br>
This code will train a edge generator training with canny and GANs.<br>
You can reuse the training edge generator that I left in (ckpts/EdgeModel_gen.pth)

## Sources and resources:
Papers with code in SISR: https://paperswithcode.com/task/image-super-resolution

Original SRGAN: https://github.com/twhui/SRGAN-PyTorch

Other SRGAN: https://github.com/kunalrdeshmukh/SRGAN

Original Edge Informed SISR: https://github.com/knazeri/edge-informed-sisr

