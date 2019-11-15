# Weekly Report 02

Date: 15/11/2019



## Things we did this week

1. We look into the paper [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104) and write a summary as the PDF file 'SIE-CNN-Summary', which can also be used in the survey part of the final report.
2. 



## Questions

### Dataset

The MNIST-Scale is introduced in [Learning Invariant Representations with Local Transformations](https://arxiv.org/pdf/1206.6418.pdf), where a scale factor is randomly sampled from the uniform distribution $U(0.3, 1)$ and they use the scale factor to randomly rescale the original MNIST dataset. We didn't find any available MNIST-Scale dataset online. A possible resource should be at [Deep Learning Datasets](http://deeplearning.net/datasets/) but the link for [MnistVariations](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations) is removed. Are we supposed to synthesize the dataset by ourselves? Though the method is not difficult to implement, with respect to the dataset size (60k+10k), it will take a lot of time. Can you give us some more efficient ways to get the scaled dataset?



Similar problem for the Fashion-MNIST-Scale.



### Replication

In 



### Experiment

The implementations of the three papers are under different frameworks: [Locally Scale-Invariant Convolutional Neural Network](https://github.com/akanazawa/si-convnet) in Caffe, [Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks](https://github.com/rghosh92/SS-CNN) in PyTorch and [Deep Scale-spaces: Equivariance Over Scale](https://github.com/deworrall92/deep-scale-spaces) in PyTorch. As explained in above, we find the first one difficult to implement and spend a lot of time on deploying the environment. In this case do we need to convert it to PyTorch in order to control variates?

Also they have their own architecture. Do we keep the structure of the network as in the paper and then compare the performance on Oral Cancer dataset? 