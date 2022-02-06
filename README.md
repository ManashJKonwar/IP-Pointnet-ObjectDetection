# Frustum Pointnet Implementation for 3D Object Detection Using TF2
> The entire repository further helps us to implement point clouds and create a custom object detecter for specially autonomous vehicle best dataset.

> Live demo [_here_](https://www.example.com). <!-- If you have the project hosted somewhere, include the link here. -->

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Dataset Utilized](#dataset-utilized)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [References](#references)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
<!-- * [License](#license) -->

## General Information
- The aim of this repository is to implement Frustum Pointnets on readily available 3d KITTI as well as Lyft datasets.
- Understand Pointnet Architecture and develop POC around research papers from main authors of Frustum Pointnet.
- Implement Frustum Pointnet for developing a 3d object detector.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->
![Frustum Pointnet 3D Object Detection Pipeline](./repo_assets/FrustumPointnet_Object_Detection_Pipeline.jpg)

## Technologies Used
- Tensorflow2
- Keras

## Features
List the ready features here:
- Preprocess KITTI Dataset - In Progress
- Train Frustum Poitnnet on KITTI Dataset - Not Yet Started
- Inference Script on KITTI Dataset - Not Yet Started
- Preprocess Lyft Dataset - In Progress
- Train Frustum Poitnnet on Lyft Dataset - Not Yet Started
- Inference Script on Lyft Dataset - Not Yet Started

## Screenshots
![Training Loss KITTI](./repo_assets/Pointnet_Classifier_Frontend.jpeg)
![Training Loss Lyft](./repo_assets/Pointnet_Part_Segmenter_Frontend.jpeg)

## Setup:
- git clone https://github.com/ManashJKonwar/IP-Pointnet-ObjectDetection.git (Clone the repository)
- python3 -m venv IPPointnetODVenv (Create virtual environment from existing python3)
- activate the "IPPointnetODVenv" (Activating the virtual environment)
- pip install -r requirements.txt (Install all required python modules)

## Dataset Utilized:
- [KITTI Dataset](http://3dvision.princeton.edu/projects/2014/3DShapeNets/)
XXXXX

- [Lyft Dataset](http://3dvision.princeton.edu/projects/2014/3DShapeNets/)
XXXXX

## Usage
### For Training PointNet:
- python train_OD.py "kitti"
- python train_OD.py "lyft"

## Project Status
Project is: __in progress_ 
<!-- / _complete_ / _no longer being worked on_. If you are no longer working on it, provide reasons why._ -->

## Room for Improvement
Room for improvement:
- Develop frontend to visualize diffrent frames from datasets
- Provide support for CPUs, GPUs and TPUs as well

To do:
- Finish KIITI dataset operation.

## References
[1] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation; Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas;
CVPR 2017; https://arxiv.org/abs/1612.00593.
[2] Frustum PointNets for 3D Object Detection from RGB-D Data; Charles R. Qi, Wei Liu, Chenxia Wu, Hao Su, Leonidas J. Guibas; https://arxiv.org/abs/1711.08488

## Acknowledgements

## Contact
Created by [@ManashJKonwar](https://github.com/ManashJKonwar) - feel free to contact me!

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
