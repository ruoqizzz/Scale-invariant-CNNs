## The summary of Locally Scale-Invariant Convolutional Neural Networks

In this paper, a model was introduced to allow the Convolution Neural Network to learn the locally scal-invariant features without increasing the number of parameters in model. 

Their scale-invariance is built in at the layer level, the difference is shown in the image below.  

![The stucture of Scale-invariant Convolution Layer](https://tva1.sinaimg.cn/large/006y8mN6gy1g8z2s7az37j30yj0ia41y.jpg)

The subimage (a) is structure of normal convolution layer and (b) is the strucutre of SI-ConvNet(Scale-invariant Convolution Layer). There are four steps in one SI-ConvNet layer. Firstly, the input image is scaled. Secondly, the same filter is applied to different scaled images. Thirdly, the feature maps are applied using inverse transformation(either cropped or padded with 0s to be properly aligned). Finally, the maxpool is applied over scaled. These steps ensure the the layer get the locally invariant features and also the same output as normal convolution layer.



They also use a tricky idea to implement SI-ConvNet layer without incraseing the number of parameters. Instead of using n scales, they increase the number of feature maps by n times because that scale the images then use the same filter to convolve is same as  using filters of different size to convole a single images. With this operation, SI-ConvNet can be trained without adding more paramters.



