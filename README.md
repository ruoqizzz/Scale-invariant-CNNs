# Scale Invariance/Equivvariance CNN


## Summary of this week

We looked through four papers this week (listed below) and we found the theory behind papers is difficult to understand and still not fully figured them out.

 

## Paper:

1. [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104)
   - **Method**: Firstly,  they applies filters at multiple scales in each layer so a single filter can detect and learn patterns at multiple scales. Then, 	max-pool responses over scales to obtain representations that are locally scale invariant yet have the same dimensionality as a traditional ConvNet layer output.
   - **Dataset**: MNIST-Scale
2. [Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks](https://arxiv.org/abs/1906.03861)
   - **Method**: Using the log-radial harmonics as a complex steerable basis, we construct a lo-
     cally scale invariant CNN, where the filters in each convolution layer are a linear combination of the basis filters.
   - **Dataset**: MNIST-Scale and FMNIST-Scale
3. [Deep Scale-spaces: Equivariance Over Scale](https://arxiv.org/abs/1905.11697)
   - **Method**: Extend convolutions to these classes of symmetry under noninvertible transformations via the theory of semigroups. Their contributions are the introduction of a semigroup equivariant correlation and a scale-equivariant CNN.
   - **Dataset**: Patch Camelyon and Cityscapes
4. [POLAR TRANSFORMER NETWORKS](https://arxiv.org/abs/1709.01889)
   - **Method**: Combines the ideas of STN and canonical coordinate representations to achieve equivariance to translations, rotations, and dila- tions.
   - **Dataset**: ROTATED MNIST, SIM2MNIST(newly introduced)



## Schedule

We decide to choose paper 1-3 to implement and generate their results. For the 3rd paper [Deep Scale-spaces: Equivariance Over Scale](https://arxiv.org/abs/1905.11697), we will use the dataset MNIST-Scale and FMNIST-Scale rather than their dataset.

**11 Nov -  08 Dec**: 

 - Finish the survery of relevant literatures
 - Implement and repeat the results of the three papers.

**09 Dec - 22 Dec**: 

- Evaluate the performance of the three papers' methods on Oral Cancer dataset (Evaluation methods haven't decided yet)
- Write the report

**07 Jan - 12 Jan**:  

- Make poster
- Write the report



 ## Work Allocation

Ruoqi:



Wei: