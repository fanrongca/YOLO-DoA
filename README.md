# YOLO-DoA
# Introduction

YOLO-DoA is an efficient YOLOv3-based approach for DoA estimation, which is implemented as a regression task to spatially separated angular boxes. DoAs of sources with confidence scores
are directly predicted from the spectrum proxy with YOLO-DoA and an end-to-end estimation is realized. 
Simulation results demonstrate that the proposed approach outperforms several state-of-the-art methods in terms of network size, computational cost,prediction time and accuracy of DoA estimation.

TABLE I: The effectiveness study of YOLO-DoA. MPS represents the mini-batch per second.

| |Methods |Parameters |GFLOPs |MPS |RMSE |
|--- |---  |---  |---    |---    |---    |
|A|YOLO-Basic|22.391M|81.041|1.74|1.9°,6.3°|
|B|YOLO-ResNet18|5.496M|18.649|3.61|1.9°,7.2°|
|C|YOLO-ResNet18<sup>+|0.162M|0.721|8.22|2.3°,7.6°|
|D|+ CSP Connection|0.080M|0.332|8.53|2.2°,7.5°|
|E|+ GIoU Loss|0.080M|0.332|8.23|1.5°,6.5°|
|F|+ SE Operation|0.081M|0.333|8.08|1.4°,6.2°|
|G|+ Grid Sensitive|0.081M|0.333|8.11|1.6°,6.5°|
|H|+ SPP Layer|0.108M|0.397|8.04|1.5°,6.4°|

Through steps A → F, the construction of YOLO-DoA is completed. Compared to YOLO-Basic, both the parameters and computational cost of YOLO-DoA are reduced by 99.6%.
Meanwhile, the prediction speed is increased by a factor of 4.6 and RMSE is decreased obviously. Therefore, the effectiveness
of YOLO-DoA is confirmed. Moreover, the Grid Sensitive and Spatial Pyramid Pooling(SPP) layer are additionally tested in the experiment. 
The results show that these two modules will deteriorate performance of DoA estimation, hence they are not adopted in YOLO-DoA.

# Updates
- 【2022/01/19】We upload the source code of YOLO-DoA model
- 【2022/01/20】We upload test files and prediction code
  
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
- tools.py -- Auxiliary function, including postprocess of predicted boxes,soft-NMS
- computationcost.py-- Parameter statistics and  computational cost statistics function
- modules.py -- Implementation functions of CBL and Ups
- backbone.py -- Backbone of YOLO-DoA
- yolovdoa_test.py -- Be responsible for reading the trained model and test file, outputting the predicted angle and calculating the RMSE
- Test.tfrecord -- A demonstration test file containing sample and incident directions
  
# Test step
- (1)open the yolovdoa_test.py
- (2)In Line 30, modify the "test_path" to the full path of the test tfrecord file (ie. Test.tfrecord
- (3)In Line 31, modify the "saved_model_dir" to the full path where the trained model file (ie. saved_model.pb) is located
- (4)Run the yolovdoa_test.py
- (5)The console will print RMSE at scene ± 85 ° and scene ± 90 °
- (6)CSV file (ie. predict.csv) containing real and predicted angels is generated in "saved_model_dir" and can be used to analyze the prediction  
# Contact
Issues should be raised directly in the repository. For professional support requests please email Rong Fan at fanrong@cafuc.edu.cn.

  
  
  
  
  
  
