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

## Analyzing the challenge:





Original SRGAN: https://github.com/twhui/SRGAN-PyTorch

Other SRGAN: https://github.com/kunalrdeshmukh/SRGAN

Original Edge Informed SISR: https://github.com/knazeri/edge-informed-sisr
