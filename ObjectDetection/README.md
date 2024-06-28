# ComputerVision - Object Detection


## Convolution Neural Networks

* CNNs use a Convolution layer with a specific kernal size and stride length

* A convolution layer is generally followed by activation function and pooling layer.

* Pooling can be Max Pooling or Average Pooling. This is mainly done to reuce the image.

* These are followed by fully connected layers

### Popular Architectures

* **LeNet**: This was first developed by Yann LeCun in 90s.

* **AlexNet**: AlexNet had similar architecture to LeNet, but was much deeper. This featured convolution layers stacked on top of each other in 2012.

<center>

![alt text](images/alexnet.png)
*AlexNet Architecture*

</center>


*  **GoogleLeNet**: This has introducted inception module, which reduced the number of parameters. Additionally the work uses average pooling to eliminate a much larger paremters in FC layers. Most recent one is inception-V4

<center>

![alt text](images/inception.png)
*Inception-V4 Architecture*

</center>

* **VGGNet**: This work has shown that depth is a crucial parmameter and uses 3x3 and 3x3 pooling from begining to end. This uses a lot of memory and has around 140M parmeters.


<center>

![alt text](images/vgg16.jpg)
*VGG16 Architecture*

![alt text](images/vgg16summary.jpg)
*VGG16 Summary*

</center>

* **ResNet**: This uses residual connections to improve the overall performance of the image processing tasks. This is a default choice for any image task


<center>

![alt text](images/resnet.webp)
*ResNet Architecture*


</center>


## Region Based Convolutional Neural Networks (RCNN)

<div align="center">

| CNN | RCNN |
|:---------------------------------------------------------:|:--------------------------------------------------------------------------:|
|Generic Algorithm    | Uses Two Stage detection using selective search  |
| Mostly useful for single object classification   | Generates potential local areas to identify objects   |
| Needs redundant processing   | Uses shared feature extraction  |


</div>

### RCNN Familiy

* There are four major architecture in RCNN family
    1. R-CNN
    2. Fast R-CNN
    3. Faster R-CNN
    4. Masked R-CNN
 
### Architectures

* **RCNN**: Generates and Extracts category independent regions such as candidate 
bounding boxes
    * RCNN has three major modules
        * The first module extracts region proposals to identify candidate modules. This module generates areound 2k region proposals for feature extaction.
            1.  In selective search the first step is to generate initial sub-segmentation images
            2. Recursively combines through greedy approach
            3. Uses the generated regions to produce candidate object location

        * Second module computes CNN features. This is the selective search algorithm
            1. Extract fixed length vectors from each region from first stage
            2. This can be high dimensional vector such as 4096 from each proposed regions (internally uses VGG16)

        
        * Third Module is the classifier
            1. The extracted data is fed to a linear SVM to predict the classes
            2. Also performs bound box regression to generate the bound box (x,y, width and height)
    
    * RCNN uses multi-stage pipeline which can be slow
    * Training is expensive 
    * Deterction is also slow
        


<center>


![alt text](images/RCNN-architecture.png)
*RCNN Architecture*

</center>

* **Fast RCNN**: Here the three independent modules of RCNN are combined to improve the computation speed and also uses shared computation results.
    * The main contribution is the roi pooling layer
        1. Here the input image is divided into windows
        2. The input image is dvivied in HxW grids
        3. RoI get a fixed size feature from each region
    * CNN extracts a high level features from entire feature image and generates a feature map which is processed by severeal convolution and pooling layers
    * So in summary, the deep ConvNet processes the while input image to generate conv feature map, which is then processed with RoI pooling layer to generate fixed length vectors by diving proposed region into grids and picking up the most representative regions
    * The RoI pool is then given to two branches
        1. Classification to identify the image
        2. Bounding box regression to localize the image
    * Fast RCNN has higher mean and average accuracy then RCNN
    * Although, this is faster it still involves multiple networks to extract the RoIs

<center>

![alt text](images/FastRCNN.jpg)
*Fast RCNN Architecture*

</center>

* **Faster RCNN**: To  speed up, one way was to integrate Region Proposal Network into the RCNN to accelerate the inference time
    * This has two modules which operate on output of VGG network
        1. Region Proposal Network (RPN)
        2. Fast-RCNN
    * RPN acts a attetnion mechanism for Fast RCNN, this operates on the output feature map of the backbone CNN by sliding and anchor
    * This has the highest mean average precision


<center>

![alt text](images/FasterRCNN.png)
*Faster RCNN Architecture*

</center>

* **Mask RCNN**: This is an extension of Faster RCNN.
    * This can also perform instance segmentation
    This decouples classification and mask prediction.
    * Mask branch is fully connected network applied to each region of interest, predicting a segmentation mask in pixel to pixel manner
    * This has RoI Align, instead of RoI polling and a similar RPN whcih uses feature maps from Deep CNN layer
    * RoI Aligns more pixel level information than pooling.



<center>

![alt text](images/overview_rcnn.png)
*Overview of RCNN Architectures*




</center>


    

## References

* https://www.mdpi.com/2072-4292/9/8/848
* https://www.mdpi.com/2072-4292/13/22/4712
* https://www.kaggle.com/discussions/getting-started/178568
* https://idiotdeveloper.com/vgg16-unet-implementation-in-tensorflow/
* https://medium.com/@siddheshb008/resnet-architecture-explained-47309ea9283d
* https://www.researchgate.net/figure/RCNN-architecture-17_fig4_341099304
* https://towardsdatascience.com/fast-r-cnn-for-object-detection-a-technical-summary-a0ff94faa022