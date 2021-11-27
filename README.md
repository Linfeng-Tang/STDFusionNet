

# STDFusionNet

The code of "STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection"
## Architecture
![The architecture of the proposed infrared and visible image fusion network based on the salient target detection. The mask is only needed to construct
loss function in the training of the model, and is not needed in the testing phase.](https://github.com/Linfeng-Tang/STDFusionNet/blob/main/Figure/Architecture.png)

## To Train

Run "**CUDA_VISIBLE_DEVICES=0 python train.py**" to train the network.

## To Test

Run "**CUDA_VISIBLE_DEVICES=0 python test_one_image.py**" to test the network.

## Recommended Environment

 

 - [ ] List item
 - [ ] python 3.6.0
 - [ ] TensorFlow-GPU 1.14.0
 - [ ] scipy 1.2.0
 - [ ] OpenCV 3.4.2
 - [ ] numpy 1.19.2

## If this work is helpful to you, please cite it as：
```
@article{ma2021STDFusionNet,
  title={STDFusionNet: An Infrared and Visible Image Fusion Network Based on Salient Target Detection},
  author={Jiayi Ma, Linfeng Tang, Meilong Xu, Hao Zhang, and Guobao Xiao},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021},
  volume={70},
  number={},
  pages={1-13},
  doi={10.1109/TIM.2021.3075747}，
  publisher={IEEE}
}
```
