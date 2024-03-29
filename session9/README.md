# NetD CNN Model for Image Classification

## Overview
The NetD model is a convolutional neural network (CNN) designed for efficient image classification. Achieving a high accuracy of 61.26% with only 100K parameters, the model demonstrates effective feature extraction and generalization capabilities. The model is structured to have a total receptive field (RF) of 47, adhering to the design requirement of an RF greater than 44.

## Model Architecture
The NetD model is crafted with the following key features:
- **Total Receptive Field (RF) > 44**: Ensuring broad context understanding with an actual RF of 47.
- **Depthwise Separable Convolution**: Utilized in one of the layers to efficiently increase the channel depth while controlling the parameter count.
- **Dilated Convolution**: Incorporated to expand the receptive field without increasing the spatial dimensions or parameter size significantly.
- **Global Average Pooling (GAP)**: Applied before the final fully connected (FC) layer to reduce spatial dimensions to 1x1, focusing on the most relevant features.
- **Fully Connected Layer**: Tailored to the number of target classes, processing the output from the GAP to make the final class predictions.

### Specific Layer Configurations
- **ConvBlock1**: Standard convolution with stride to initiate feature extraction.
- **ConvBlock2**: Depthwise Separable Convolution to enhance feature depth efficiently.
- **ConvBlock3**: Dilated Convolution to increase the receptive field effectively.
- **ConvBlock4**: Concludes the convolutional stages, using stride of 2 and Dilated kernels as `nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2, padding=2, dilation=2)`
## Data Augmentation
Utilizing the Albumentation library, the following transformations are applied during training to improve model robustness and generalization:
- **Horizontal Flip**: To introduce variability in the horizontal axis.
- **ShiftScaleRotate**: To mimic object variations in position, size, and orientation.
- **CoarseDropout**: To simulate occlusion and robustness against missing parts, with parameters adjusted to fill with dataset mean color values.

## Performance and Future Work
- **Current Accuracy**: Reached 61.26% on the validation set, demonstrating the model's effectiveness.
- **Parameter Efficiency**: With only 100K parameters, the model is lightweight yet performs robustly.
- **Potential Improvements**: Given more time for model tuning and expanded data augmentation, there's potential to achieve even higher accuracy.

## Conclusion
NetD exemplifies a compact yet effective CNN architecture with strategic layer choices and augmentation techniques, aligning with specific design goals and performance metrics. Future iterations of the model can explore more extensive hyperparameter tuning and advanced augmentation strategies to further enhance accuracy and generalizability.
