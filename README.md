# OVRM
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

By [Menghan Hu](https://faculty.ecnu.edu.cn/_s15/hmh/main.psp), Xinxin Zhang

If you have any questions, please contact Menghan Hu(mhhu@ce.ecnu.edu.cn) or Xinxin Zhang(Zhangxinxin5058@163.com).

## A Gentle Introduction
The OVRM(Optimizing Video-based Respiration Monitoring) algorithm reduces motion artifacts through adaptive thresholding and dynamic ratio mechanisms, and selects the optimal respiratory ROIs by evaluating periodicity, similarity, smoothness, and energy, effectively enhancing the accuracy of respiratory monitoring.

This is a overview of the peak-trough adaptive motion artifact removal method.
![image](https://github.com/zxx5058/OVRM/blob/main/ImagesFolderForReadMe/Motion_artifact.png)

This is a overview of the Characteristic-Driven Adaptive ROI Selection method.
![image](https://github.com/zxx5058/OVRM/blob/main/ImagesFolderForReadMe/ROI_selection.png)

## Experiment result
Bland-Altman plots (up) and correlation plots (down) of estimated rate among different datasets.
![image](https://github.com/zxx5058/OVRM/blob/main/ImagesFolderForReadMe/BA.png)
