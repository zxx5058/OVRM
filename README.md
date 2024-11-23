# Optimizing Video-based Respiration Monitoring: Motion Artifact Reduction and Adaptive ROI Selection (OVRM)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

By Xinxin Zhang, [Menghan Hu](https://faculty.ecnu.edu.cn/_s15/hmh/main.psp)

If you have any questions, please contact Xinxin Zhang(Zhangxinxin5058@163.com) or Menghan Hu(mhhu@ce.ecnu.edu.cn).

## 🌟Dataset
The Motion dataset and the Static dataset respectively consist of 201 videos, each of 30 seconds duration, which can be used in a non-contact respiratory signal detection task, with their corresponding true respiratory signals in the corresponding folder.

## 🌟A Gentle Introduction
The OVRM(Optimizing Video-based Respiration Monitoring) algorithm reduces motion artifacts through adaptive thresholding and dynamic ratio mechanisms, and selects the optimal respiratory ROIs by evaluating periodicity, similarity, smoothness, and energy, effectively enhancing the accuracy of respiratory monitoring.

This is an overview of the peak-trough adaptive motion artifact removal method.
![image](https://github.com/zxx5058/OVRM/blob/main/ImagesFolderForReadMe/Motion_artifact.png)

This is an overview of the Characteristic-Driven Adaptive ROI Selection method.(https://github.com/zxx5058/OVRM/blob/main/ImagesFolderForReadMe/ROI%20Selection%20Module_00.png?raw=true)

## 🌟Experiment result
Bland-Altman plots (up) and correlation plots (down) of estimated rate among different datasets.
![image](https://github.com/zxx5058/OVRM/blob/main/ImagesFolderForReadMe/BA.png)


