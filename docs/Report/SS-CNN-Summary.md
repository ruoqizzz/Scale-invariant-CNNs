## Summary of Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks

In this paper, the authors proposed a scale-steerable filter and introduced a scale-steered kernel based on a linear combination of the scale-steerable filters. Then the Scale-Steered CNN is proposed by replacing the kernels in [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104) with the scale-steered kernels. 



Based on the rotation steerable circular harmonics from [Harmonic Networks: Deep Translation and Rotation Equivariance](https://arxiv.org/abs/1612.04642), the mathematical expression of a scale-steerable filter is, 
$$
S^{kj}(r,\phi)=\frac{1}{r^m}(K(\phi,\phi_j)+K(\phi,\phi_j+\pi))e^{i(k(\text{log }r)+\beta)},
$$
where $r$ and $\phi$ are the radius and angle in polar co-ordinates, $k$ is the filter order, $\phi_j$ is the mean of Gaussian form $\phi$,  $K(\phi,\phi_j)=e^{-d(\phi,\phi_j)^2/{2\sigma_\phi^2}}$ with $d(\phi,\phi_j)$ equal to the distance between $\phi$ and $\phi_j$, and $\beta$ is a phase offset term. Note that the expression is complex-valued, which means it has real and imaginary parts. Examples of scale-steerable basis filters are shown below:

<img src="https://tva1.sinaimg.cn/large/006tNbRwgy1g9hsx62k42j30x80gcn0b.jpg" alt="Scale-steerable basis filters" style="zoom:45%;" />



Then they proposed and proved a theorem. Assume in a large image a circular patch $I(a)$ is the range of $x,y$ with $0 \le \sqrt{x^2+y^2} \le a$, and $I^s(a)$ is the same patch when the image was scaled around the center of the patch by a scale factor $s$. Then the following equation holds:
$$
[I^s(a) \star S^{kj}(a)]=s^{m-2}e^{-i(k \text{ log }s)}[I(as) \star S^{kj}(as)],
$$
where $\star$ is the cross-correlation operator. When $a=\infty$, $[I^s \star S^{kj}]=s^{m-2}e^{i(k \text{ log }s)}[I \star S^{kj}]$.



A scale-steered kernel $W$ consisting of a linear combination of the steerable basis filters is defined as $W=\sum_{k,j}c_{kj}S^{kj}$, where $c_{kj}$ is constant. One of the property the scale-steered kernel $W$ has is scale steerability, giving
$$
W^s(as)=s^{m-2}\sum_k e^{-ik \text{ log }s}(\sum_j c_{kj}S^{kj}(as)).
$$
Note that only the real part of the scale-steered kernel is used.



The corresponding Scale-Steered CNN is to replace the kernels in [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104) with scaled scale-steered kernels from the above equation. The proposed scale-invariant layer with scale-steered filters is shown below:

![Scale-invariant layer](https://tva1.sinaimg.cn/large/006tNbRwgy1g9hu0pzy5rj30zu0hc0wo.jpg)



For the experiments, they evaluated their model on both global scale datasets (MNIST-Scale and FMNIST-Scale) and local scale dataset (MNIST-scale-local-2). The global scaled datasets were created by a random scale factor $s\in(0.3,1)$, and the local scaled variations were implemented through scaling pairs of MNIST samples with $s\in(0.7,1)$ and arranging them side by side to size $28 \times 40$. They evaluted the model performance from the following aspects:

- Global scale variations: MNIST and FMNIST. They split the dataset into 10k, 2k, and 50k into training, validation and testing data to test the generalization ability of the model. The network structure contained 3 convolutional layers and 2 fully connected layers. They compared the mean and standard deviation of the test error over 6 splits.
- Generalization to Distortions. They trained the network on the undistorted MNIST-Scale and tested on MNIST-Scale with added elastic deformations. They compared the best performing test-time with respect to different elastic distortion extents.
- Synthesized data: Local scale variations. They compared the mean and standard deviation of the test error over 6 splits with respect to different sizes of training data.
- Visualization Experiments. They visualized the first layer filters and the averge feature map activation of the first layer to compare the ability of structure preserving.