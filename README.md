# Overview

![](./images/overview.png)

# Image segmentation model

We can grab something from the web.

- [TF tutorial](https://www.tensorflow.org/tutorials/images/segmentation)
- [Berkeley dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- [Segment anything](https://ai.facebook.com/datasets/segment-anything-downloads/)
- [Masked R-CNN](https://github.com/matterport/Mask_RCNN)

Data: needs to have labels.

# Image in-painting model

We have to train this.

- [DeepFill v1, v2](https://mmediting.readthedocs.io/en/v0.12.0/inpainting_models.html)
- [PConv](https://github.com/MathiasGruber/PConv-Keras)
  - [Origial paper](https://arxiv.org/pdf/1804.07723v2.pdf)
  - [PConv layer explained](https://towardsdatascience.com/pushing-the-limits-of-deep-image-inpainting-using-partial-convolutions-ed5520775ab4)
  - Keeps track of mask (1 if valid pixel is involved in convlution op, 0 otherwise)
  - Per-pixel loss (hole/valid)
  - Perceptual loss (VGG loss) - should have similar rep. as vgg16 (pool1, pool2, pool3 only)
  - Style loss - utilizes VGG16 generated gram matrix.
  - Total variation loss: ensures smoothness
- [Image in-painting tutorial](https://wandb.ai/ayush-thakur/image-impainting/reports/An-Introduction-to-Image-Inpainting-Using-Deep-Learning--Vmlldzo3NDU0Nw)

# Roles

- Setup image segmentation
  - Inference: Alekhya
- Setup image inpainting
  - Train: Jiyoon
  - Inference: Insoo
