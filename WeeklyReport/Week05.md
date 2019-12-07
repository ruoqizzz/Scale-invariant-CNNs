# Weekly Report 05
## Things we did this week
- Read SS-CNN related papers
- Evaluated the SS-CNN (details in file SS-CNN-Implementation.pdf)
- Read new papers in translation equivariance/invairance erea
## Our ideas for the choice of 3rd paper
In this week, we found a paper [*Making Convolutional Network Shift-Invariant Again*](https://arxiv.org/abs/1904.11486) which points out that modern convolutional networks are not shift-invariant and classic anti-aliasing can be integrated with existing downsampling to improve shift-equivariance of deep networks. They validated on max-pooling, average-pooling, strided-convolution in different architectures and the result shows the effective regularization and better generalization.

When reading the first two papers (SI-ConvNet and SS-CNN), we found that the all the authors assume CNNs is by definition translation equivariant but it is proved not. Also shift-invariance/equivariance is the basis of scale-invariance. In other words, without shift-invariance/equivariance, even a perfect scale-invariance/equivariance network would not get great performance because after scaling the object in the image would have different location.

Therefore, we want to combine the methods in paper [*Making Covolutional Network Shift-Invariant Again*](https://arxiv.org/abs/1904.11486) with SS-CNN and experiment these two networks in the next two weeks. 


## Updated Schedule


- **11 Nov - 24 Nov**:
    - Write the summary of *Locally Scale-Invariant Convolutional Neural Network*. 
    - Implement the results of *Locally Scale-Invariant Convolutional Neural Network on MNIST-Scale dataset*.

- **25 Nov - 08 Dec**:
    - Write the summary of *Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks*.
    - Implement the results of *Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks* on MNIST-Scale.

- **09 Dec - 22 Dec**:
    - Write the summary of *Making Convolutional Network Shift-Invatiant Again*
    - Combine the method with SS-CNN, denoted as SS-CNN-BlurPool
    - Evaluate the method on MNIST-Scale.
    - Implement the baseline CNN on MNIST-Scale
    - Compare the results of CNN, SS-CNN, SI-ConvNet, and SS-CNN-BlurPool.
- **06 Jan - 12 Jan**:
    - Write the report. 
    - Design poster.
    
## Our Questions
- Still confused with *Log-Radial Harmonics: A scale-steerable filter basis*. We don't understand the theory but know how to use these filters.
- We now change the third paper to *Making Convolutional Network Shift-Invariant Again*. But this paper improves the CNN in a general way rather than focusing on Scale-Invariance/Equivariance. We don't know whether it is a good change but we believe there is a huge connection between shift-invariance and scale-invariance/equivariance. Also the new method can improve the result of any scale-invairance/equivariance network.