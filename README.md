# General CNN Models for Image Recognition Job
A template which is suitable for all kinds of image recognition job

## Version:
+ V 1.1

## Networks included:
+ MobileNet_V1
+ MobileNet_V2
+ MobileNet_V3
+ EfficientNet
+ ResNeXt
+ InceptionV4, InceptionResNetV1, InceptionResNetV2
+ SE_ResNet_50, SE_ResNet_101, SE_ResNet_152
+ SqueezeNet
+ DenseNet
+ ShuffleNetV2
+ ResNet


## Preparation
+ Requirements:
    + Python >= 3.6
    + Tensorflow == 2.1.0


## Training Steps
+ Data Input Preparation:
    + Prepare image data, and saved under following folder structure
    + Run the script **python split_dataset.py** in shell/ps/cmd to split the raw dataset into train set, valid set and test set
    + Run **python to_tfrecord.py** in shell/ps/cmd to generate tfrecord files<br/>

```
Before
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```

```
After
|——dataset
   |——train
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |——valid
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |—-test
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
```

+ Configurations
    + Change the corresponding parameters in **config.py**.
        + Finetune **model_index** to select the corresponded model
        + Finetune **hyper-parameter** in order to make better convegence performance and fitting quality
        + Finetune the **threshold** in order to setup model saving accuracy level(Under this level, model will not be saved)
        + Finetune the **training loop settings** in order to find suitable condition for CPU/MEM for your environment
        + Finetune the **training input settings**
            + NUM_CLASSES must be the same as your classes defined
            + IMAGE_HEIGHT, IMAGE_WIDTH must be the same as following tables, see section **Different input image sizes for different neural networks**   
+ Run **python train.py job_id product_id** in shell/ps/cmd to start training.<br/>
+ See the Log folder for model training performance benchmark offline if needed
 ```
|——log
    |——job_id
        |——product_id
            |——training_result.log
            |——training_result_step.log
 ```

## Evaluate(Testing) Steps
+ Run **python evaluate.py job_id product_id** in shell/ps/cmd to evaluate the model's performance on the test dataset
+ See the Log folder for model training performance benchmark offline if needed
|——log
    |——job_id
        |——product_id
            |——test_result.log
            |——training_result_step.log

## Prediction Steps(Temporary):
+ Put the data(should be image format) under following folder:
|——production_dataset
+ Run **python predict.py job_id product_id** in shell/ps/cmd to get the model's output class, currently, it will be print in the window.


## Different input image sizes for different neural networks
<table>
     <tr align="center">
          <th>Type</th>
          <th>Neural Network</th>
          <th>Input Image Size (height * width)</th>
     </tr>
     <tr align="center">
          <td rowspan="3">MobileNet</td>
          <td>MobileNet_V1</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>MobileNet_V2</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>MobileNet_V3</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>EfficientNet</td>
          <td>EfficientNet(B0~B7)</td>
          <td>/</td>
     </tr>
     <tr align="center">
          <td rowspan="2">ResNeXt</td>
          <td>ResNeXt50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNeXt101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="3">Inception</td>
          <td>InceptionV4</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td>Inception_ResNet_V1</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td>Inception_ResNet_V2</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td rowspan="3">SE_ResNet</td>
          <td>SE_ResNet_50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SE_ResNet_101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SE_ResNet_152</td>
          <td>(224 * 224)</td>
     </tr>
     </tr align="center">
          <td>SqueezeNet</td>
          <td align="center">SqueezeNet</td>
          <td align="center">(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="4">DenseNet</td>
          <td>DenseNet_121</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_169</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_201</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_269</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ShuffleNetV2</td>
          <td>ShuffleNetV2</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="5">ResNet</td>
          <td>ResNet_18</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_34</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_152</td>
          <td>(224 * 224)</td>
     </tr>
</table>

## Future Plan(V1.2)
+ Change Estimated:
    + Upgrade evaluation script for auto iteration for testing all models
    + Upgrade evaluation script for judgement of best model inside all epochs
    + Upgrade evaluation script for generating analysis kits for training performance evaluation
    + Upgrade production script for full-auto folder scaning for multiple picture handling
+ Release Schedule:
    + 2020-04-13

## References
+ MobileNet_V1: [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
+ MobileNet_V2: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
+ MobileNet_V3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
+ EfficientNet: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
+ The official code of EfficientNet: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
+ ResNeXt: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)
+ Inception_V4/Inception_ResNet_V1/Inception_ResNet_V2: [Inception-v4,  Inception-ResNet and the Impact of Residual Connectionson Learning](https://arxiv.org/abs/1602.07261)
+ The official implementation of Inception_V4: https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py
+ The official implementation of Inception_ResNet_V2: https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
+ SENet: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
+ SqueezeNet: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
+ DenseNet: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
+ https://zhuanlan.zhihu.com/p/37189203
+ ShuffleNetV2: [ShuffleNet V2: Practical Guidelines for Eﬃcient CNN Architecture Design
](https://arxiv.org/abs/1807.11164)
+ https://zhuanlan.zhihu.com/p/48261931
+ ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)