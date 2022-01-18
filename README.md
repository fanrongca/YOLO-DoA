# YOLO-DoA
# Introduction

YOLO-DoA is an efficient YOLOv3-based approach for DoA estimation, which is implemented as a regression task to spatially separated angular boxes. DoAs of sources with confidence scores
are directly predicted from the spectrum proxy with YOLO-DoA and an end-to-end estimation is realized. 
Simulation results demonstrate that the proposed approach outperforms several state-of-the-art methods in terms of network size, computational cost,prediction time and accuracy of DoA estimation.

TABLE I: The effectiveness study of YOLO-DoA. MPS represents mini-batch per second.

| |Methods |Parameters |GFLOPs |MPS |RMSE |
|--- |---  |---  |---    |---    |---    |
|A|YOLO-Basic|22.391M|81.041|1.74|1.9°,6.3°|
|B|YOLO-ResNet18|5.496M|18.649|3.61|1.9°,7.2°|
|C|YOLO-ResNet18<sup>+|0.162M|0.721|8.22|2.3°,7.6°|
|D|+ CSP Connection|0.080M|0.332|8.53|2.2°,7.5°|
|E|+ GIoU Loss|0.080M|0.332|8.23|1.5°,6.5°|
|F|+ SE Operation|0.081M|0.333|8.08|1.4°,6.2°|
|G|+ Grid Sensitive|0.081M|0.333|8.08|1.6°,6.5°|
|H|+ SPP Layer|0.108M|0.397|7.62|1.5°,6.4°|

Through steps A → F, the construction of YOLO-DoA is completed. Compared to YOLO-Basic, both the parameters and computational cost of YOLO-DoA are reduced by 99.6%.
Meanwhile, the prediction speed is increased by a factor of 4.6 and RMSE is decreased obviously. Therefore, the effectiveness
of YOLO-DoA is confirmed.Moreover, the Spatial Pyramid Pooling(SPP) and Grid Sensitive are additionally tested in the experiment. 
The results show that these two modules will deteriorate performance of DoA estimation, hence they are not adopted in YOLO-DoA.

# Updates
- 【2022/01/18】We upload the source code of YOLO-DoA model
- 【2022/01/19】We upload test files and prediction codes
  
# Environments

- python 3.8.6

- PyCharm Community 2018.3.2

- CUDA 10.0

- NVIDIA GeForce RTX2080
  
- Two Intel Xeon E5-2678v3 @2.50GHz CPUs and 128GB RAM

# Requirements

- h5py 2.10.0

- numpy 1.19.3

- tensorflow-gpu 1.13.1

# File description
- yolovdoa_train.py -- Data preprocessing, model training, trained model storage function

- yolovdoa.py -- Main function of YOLO-DoA, including Neck, Head sturcture, and loss calculation
- utils.py -- Auxiliary function, including postprocess of predicted boxes,soft-NMS
- computationcost.py-- Parameter statistics and calculation statistics function
- common.py -- Implementation functions of CBL and Ups
- backbone.py -- Backbone of YOLO-DoA
# Contact
Issues should be raised directly in the repository. For professional support requests please email Rong Fan at fanrong@cafuc.edu.cn.

  
  
  
  
  
  
