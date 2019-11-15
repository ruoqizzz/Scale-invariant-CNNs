## Summary of Locally Scale-Invariant Convolutional Neural Networks

In this paper, a model was introduced to allow the Convolution Neural Network to learn the locally scale-invariant features without increasing the number of parameters in the model. 

Their scale-invariance is built in at the layer level, the difference is shown in the image below.  

![The stucture of Scale-invariant Convolution Layer](https://tva1.sinaimg.cn/large/006y8mN6gy1g8z2s7az37j30yj0ia41y.jpg)

The sub-image (a) is the structure of a normal convolution layer and (b) is the strucutre of SI-ConvNet (Scale-Invariant Convolution Layer). There are four steps in one SI-ConvNet layer. Firstly, the input image is scaled. Secondly, the same filter is applied to different scaled images. Thirdly, the feature maps are applied using inverse transformation (either cropped or padded with 0s to be properly aligned). Finally, the maxpool is applied over scaled. These steps ensure the the layer get the locally invariant features and also the same output as normal convolution layer.



They also use a tricky idea to implement SI-ConvNet layer without increasing the number of parameters. Instead of using $n$ scales, they increase the number of feature maps by $n$ times because that scale the images then use the same filter to convolve is as same as using filters of different size to convole a single image. With this operation, SI-ConvNet can be trained without adding more paramters.



For the experiments, they implement SI-ConvNet in the Caffe framework and make a comparison with other neural networks on the same dataset MNIST-Scale. The architectures are composed of two convolutional layers, a fully connected layer and a soft-max logistic regression layer. They compare the performance from the following aspects:

- Test error. With the training dataset of 10k and test dataset of 50k, they compare the average test error and standard deviation over 6 training and test data folds.
- Measuring invariance. For each neuron $h_i$, they compute the ratio of invariance and selectivity $L_i/G_i$ where $G_i=\sum|h_i(x)>t_i|/N$ with a chosen $t_i$ and $L_i$ is the local firing rate.
- Effect of training data and number of parameters. They compare the test error vs the number of feature maps in the two convolutional layers and the test error vs the amount of training data.
- Robustness to unfamiliar scales and scale variation. This time they scale the training data using a scale factor from Gaussian distribution and compare the test error on different test scale. They also compare the test error vs different range of uniform distribution used to scale training data.