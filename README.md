# Scale Invariance/Equavariance Convolutional Neural Network 

This project is focus on evaluating two to three recent approaches to achive scale equivariance and/or invariance of CNNs.

## Paper:

1. [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104)
   - **Method**: Firstly,  they applies filters at multiple scales in each layer so a single filter can detect and learn patterns at multiple scales. Then, 	max-pool responses over scales to obtain representations that are locally scale invariant yet have the same dimensionality as a traditional ConvNet layer output.
   - **Dataset**: MNIST-Scale
2. [Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks](https://arxiv.org/abs/1906.03861)
   - **Method**: Using the log-radial harmonics as a complex steerable basis, we construct a lo-
     cally scale invariant CNN, where the filters in each convolution layer are a linear combination of the basis filters.
   - **Dataset**: MNIST-Scale
3.  [Making Convolutional Network Shift-Invariant Again](https://arxiv.org/abs/1904.11486)
      - **Method**: Antialiasing filter combined with subsampling, for example, max pooling and CNN with stride. 
      - **Dataset**: MNIST-Scale

 

## Schedule

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
- **23 Dec - -5 Jan**:
   - Preproccessing with dataset Oral Cancer
   - Evaluation on different training size
- **06 Jan - 12 Jan**:
    - Write the report. 
    - Design poster.
    
