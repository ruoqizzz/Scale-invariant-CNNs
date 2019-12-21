## Summary of Making Convolutional Networks Shift-Invariant Again

This paper points out that modern convolutional networks are not shift-invariant and classic anti-aliasing can be integrated with existing downsampling to improve shift-equivariance of deep networks. They validated on max-pooling, average-pooling, strided-convolution in different architectures and the result shows the effective regularization and better generalization.

Some methods often used in neural network for reducing spatial resolution such as max-pooling, average pooling and strided-convolution. Their imporvement is shown in below:

![Anti-aliasing common downsampling layers](https://tva1.sinaimg.cn/large/006tNbRwgy1ga3sr8hfhrj30wp05r3zq.jpg)

- **MaxPool** to **MaxBlurPool**

  For the simple signal(`[0,0,1,1,0,0,1,1]`) shown in the left of image below, the result of maxpooling(kernel size is 2 and stride is 2) is `[0,1,0,1]`. However, after a simply shifiting, the result would make a big difference which is `[1,1,1,1]`. Therefore, the shift-equivariance is lost. 

  ![Illustrative 1-D exmaple of sensitivity to shifts](/Users/wsgdrfz/Library/Application Support/typora-user-images/image-20191221100002152.png)

  But actually, max operation preserves shift-equivaraince. It is the subsequent subsamling lose the shift-equivaraince. Therefore, the solution is fistly, divide the Max-pooling into two functions: $MaxPool_(k,s) = Subsample_s \circ Max_k$then add an anti-aliasing filter. The operations are shown below:

  ![image-20191221095700024](https://tva1.sinaimg.cn/large/006tNbRwgy1ga4g5jxqm9j31d00os7an.jpg)

  During the implementation, blurring ans -subsampling are combined, as commonplace in image processing. It as called $BlurPool_{m,s}$. 

  $$MaxPool_{k,s} \rightarrow Subsample_s \circ Blur_m \circ Max_k=BlurPool_{m,s} \circ Max_k$$ 

- **StridedConv** to **ConvBlurPool**

  Strided-convolutions have same issue with MaxPool and their solution is same:

  $$Relu \circ Conv_{k,s} \rightarrow BlurPool_m \circ Relu \circ Conv_{k,}$$

- **AveragePool** to **BlurPool**

  Average pooling is basically same with blurred downsampling. Here, only the filter is replaced to better preserve the shift-equivriance.

  $$AvgPool_{k,s} \rightarrow BlurPool_{m,s}$$

In the experiments, they tested filters raning from size 2 to 5 with increasing smoothing and they evaluated the result using three metrics: 

1. internal feature distance

   They dissected the progressive loss of shift-equivariance by analyze the VGG architecture internally. VGG contains 5 convolution layers followed by max-pooling with stride two and one linear classifier. As shown in the top of below image, they show the internal feature distance as a function of all possible shift-offsets($\Delta h, \Delta w$) and layers. As mentioned before, the max-pooling with kernel size $k$ a stride $s$ can be divided into two functions. In order to show the loss of shift-euiqvariance, the image below also broke the maxpool layers into two part. 

![image-20191221120639501](https://tva1.sinaimg.cn/large/006tNbRwgy1ga4jwjgvigj31js0u0u0y.jpg)

​	The bottom (b) of the image is the maps of anti-aliased VGG using MaxBlurPool. It is quite clear that the shift-equivairiance through the whole layers of networks is preserved much better than baseline VGG.

2. classification consistency 

   They tested on classification of ImageNet to show the accuracy and consistency. As shown in figure below, every architecture from ResNet18 to ResNet101improve in both direction. 

   ![image-20191221122844045](https://tva1.sinaimg.cn/large/006tNbRwgy1ga4kjeockxj310e0u0n3g.jpg)

3. generation stability

   They test the generalization capability using date set from [1] in two ways. First, they tested stability to perturbations and then tested accuracy on systematically corrupted images. The result is shown below.

   ![image-20191221181316873](/Users/wsgdrfz/Library/Application Support/typora-user-images/image-20191221181316873.png)

   Stability is measured by flip rate(mFR) which shows how often the top-1 classification chanfges on average and the dataset used is ImNet-P containing short video clips of a single image with small perturbations added. After adding anti-aliasing Bin-5, the mFR decrease 1.0% comparing to baseline reuslt of ResNet50.

   They also test more when images are corrupted. The dataset ImageNet-C is tested to explore more on corruptions. This dataset contains impulse noise, defocus and glass blur, simulated frost and fog, and various digital alterations of contrast, elastic transformation, pixelation and jpeg compression. Baseline ResNet50 has the mean error rate 60.6% while anti-alising Bin-5 preformance is 58.1% which reduces the error rate by 2.5%.

   Overall, these result shows that intergrating antialiaseing in a proper way creates a more robust and generalizable network. 



## Reference

[1] Hendrycks, D., Lee, K. and Mazeika, M., 2019. Using pre-training can improve model robustness and uncertainty. *arXiv preprint arXiv:1901.09960*.