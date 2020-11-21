# Improved VGG-16

## Introduction

In this project, I have tried to increase accuracy rate by
gathering DAG-CNN architecture and Atrous convolution pyramid.
After the models are trained, there are two different ensemble
methods I have applied. First one is common one that gets 
the most common prediction among all the models and the second
one is Logistic Regression that trains a model to predict correctly.

---
References
1. [On the use of DAG-CNN architecture for age estimation with multi-stage features fusion
](https://www.sciencedirect.com/science/article/abs/pii/S0925231218313110)
2. [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
](https://arxiv.org/abs/1406.4729)
3. [AFPNet: A 3D fully convolutional neural network with atrous-convolution feature pyramid for brain tumor segmentation via MRI images
](https://www.sciencedirect.com/science/article/abs/pii/S0925231220304847)
---

> **_NOTE:_**  Not much time was spent on hypertuning.
