# [Locally Scale-Invariant Convolutional Neurual Network Implementation based on Caffe](https://github.com/akanazawa/si-convnet)

This implementation is based on BVLC's Caffe[1] with version October 24th, 2014. 

## Caffe Architecture

Caffe can be divided into three architectures, Blobs, Layers and Networks. Everything related to data is through blob, including sav ing, communication and operation. Layer is the fundermental of model and computation. Net is resposible for integrating and connect layers.

### Blobs

Blob is the basic data structural unit of Caffe and it is actually a  four dimensional array designed to stores and communicates data. It provide a unified memory interface to hold batches of images, model parameters and the derivatives for optimization generated in backpropagtion. There are two classes of data: `data` and `diff`. `data` is the normal data in network while `diff` is the derivatives of network. Blobs can synchronize these data between CPU and GPU in a way ignoring low-level details while maintaining a high level of performance. 

### Layers

Layers are the key of Caffe and there are many computational operations in layers:

1. Convolve filters
2. Pool
3. take inner products
4. Apply nonlinearities like ReLu, sigmoid
5. Normalize
6. Load data
7. Compute loss

Every layer takes one or more blobs as input and output one or more blobs after performaing calculations. 

### Networks

The networks in Caffe is a directed acyclic graph of connected layers and a typical network starts from the data layer and ends with a loss layer. Caffe does all the bookkeeping through to ensure the correctness of forward and backward propagation. 



## Main Changes to Caffe

1. `caffe.proto`: declare parameters

2. `layer_factory.cpp`: instantiate and register the new layer

3. Layer header

4. defining layers:

   - `up_layer.cpp`
   - `downpool_layer.cpp`
   - `tiedconv_layer.cpp`
   - `ticonv_layer.cpp`

5. `util/transformation.(gpp/cpp/cu)`

6. test files

   



## Protofile to use SI-Conv Layer

![](https://tva1.sinaimg.cn/large/006y8mN6gy1g92s5sfj1ej30yl0hbtbx.jpg)

- Convolution Layer:

```
 layers {
   name: "conv1"
   type: CONVOLUTION
   bottom: "data"
   top: "conv1"
   blobs_lr: 1.
   blobs_lr: 2.
   weight_decay: 1.
   weight_decay: 0.
   convolution_param {
	 num_output: 36
	 kernel_size: 7
	 stride: 1
	 weight_filler {
	   type: "gaussian"
	   std: 0.01
	 }
	 bias_filler {
	   type: "constant"
	 }
   }
 }
```



- Scale-invariant Convolution Layer:

```
 layers {
   name: "conv1"
   type: CONVOLUTION
   bottom: "data"
   top: "conv1"
   blobs_lr: 1.
   blobs_lr: 2.
   weight_decay: 1.
   weight_decay: 0.
   convolution_param {
	 num_output: 36
	 kernel_size: 7
	 stride: 1
	 weight_filler {
	   type: "gaussian"
	   std: 0.01
	 }
	 bias_filler {
	   type: "constant"
	 }		 
   }
   transformations { scale: 0.7937 }
   transformations { scale: 1.2599 }
   transformations { scale: 1.5874 }
 }
```





## Process



## Evaluation



## Reference

[1] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S. and Darrell, T., 2014, November. Caffe: Convolutional architecture for fast feature embedding. In *Proceedings of the 22nd ACM international conference on Multimedia* (pp. 675-678). ACM.