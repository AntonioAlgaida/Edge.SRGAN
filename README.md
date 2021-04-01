# Edge.SRGAN
This repository was created in order to participate in the Hackathon organized by @SpainAI, in the computer vision challenge. The objective of this challenge was the generation of high resolution images, i.e. Single Image Super Resolution (SISR). For this, I decided to implement a solution that unifies the advantages offered by SRGAN (see https://arxiv.org/abs/1609.04802) together with those offered by the edge prediction (Edge Informed SISR) introduced in https://arxiv.org/abs/1909.05305.

## Analyzing the challenge:
In this challenge we were asked to train a system that learns to generate high resolution images from low quality images. For this, we provided a training dataset where low quality images existed, as well as the corresponding high resolution images for each of these images.
In addition, another set of low resolution test data was provided and used to evaluate the proposed solutions.
What you were asked is that for the low quality test image set, generate the high quality images.
This challenge was posed by looking for a practical application of Generative Adversarial Neural Networks (GANs) algorithms.
![alt text](https://static-01.hindawi.com/articles/mpe/volume-2020/5217429/figures/5217429.fig.002.svgz)

The metric used to evaluate the solutions was the Structural similarity index (SSIM, see https://medium.com/srm-mic/all-about-structural-similarity-index-ssim-theory-code-in-pytorch-6551b455541e.)


Original SRGAN: https://github.com/twhui/SRGAN-PyTorch

Other SRGAN: https://github.com/kunalrdeshmukh/SRGAN

Original Edge Informed SISR: https://github.com/knazeri/edge-informed-sisr
