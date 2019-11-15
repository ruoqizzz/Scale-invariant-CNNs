# Weekly Report 02

Date: 15/11/2019

## Things we did this week

1. We looked into the paper [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104) and wrote a summary as the PDF file 'SIE-CNN-Summary', which can also be used in the survey part of the final report.
2. The MNIST_scale is generated following the paper Learning Invariant Representations with Local Transformations [Learning Invariant Representations with Local Transformations](https://arxiv.org/pdf/1206.6418.pdf), where a scale factor is randomly sampled from the uniform distribution $U(0.3, 1)$ and they use the scale factor to randomly rescale the original MNIST dataset. The code with more details can be view on colab through link https://colab.research.google.com/drive/1T6hIXNlt12swcpGNq_AkD6yPcwv2vIQv .
3. We tried to run the implementation code on [Github](https://github.com/akanazawa/si-convnet). The code is based on BVLC's caffee(Oct 20th, 2014). The strange thing is, instead of using Caffe as a libray, they rewrote the original code in Caffe and then used it as library after compling. We got stuck on compling after trying almost all the methods searched online.



## Our Questions

### Replication

- In this paper, authors evaluated their model using four experiments including test error, invariance measurement, effect of training data and number of parameters and robustness to unfamiliar scales and scale variation. Do we need to replicate it as they did? Or just the test error part and invariance part?

- We don't understand why the implementation code on [Github](https://github.com/akanazawa/si-convnet) rewrite the caffe instead of using it as a library. Also, **we need help to run the code.**

- The inverse transformation method used to align the feature map is not clear. 

  > When the stride is equal to the size of the kernel, applying the inverse transformation gives direct corre- spondence between convolution outputs. When it’s not, after applying the inverse transformation, the output has to be either **cropped** or padded with 0s to be properly aligned.

  How to cropped?

### Experiment on Oral Cancer Dataset 

The implementations of the three papers are under different frameworks: [Locally Scale-Invariant Convolutional Neural Network](https://github.com/akanazawa/si-convnet) in Caffe, [Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks](https://github.com/rghosh92/SS-CNN) in PyTorch and [Deep Scale-spaces: Equivariance Over Scale](https://github.com/deworrall92/deep-scale-spaces) in PyTorch. As explained in above, we find the first one difficult to implement and spend a lot of time on deploying the environment. In this case do we need to convert it to PyTorch in order to control variates?

Also they have their own architecture. Do we keep the structure of the network as in the paper and then compare the performance on Oral Cancer dataset? 