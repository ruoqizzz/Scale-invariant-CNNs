# [Locally Scale-Invariant Convolutional Neurual Network Implementation based on Caffe](https://github.com/akanazawa/si-convnet)

This implementation is based on BVLC's Caffe [1] with version of October 24th, 2014. 



## Caffe Environment Deployment

We use the cloud server with a GPU of GTX 1080 Ti 40 core on [jikecloud.net](https://www.jikecloud.net). The original code on [si-convnet](https://github.com/akanazawa/si-convnet) has some problem to run due to the update of libraries and we change the code a little bit. The updated code is on [our github](https://github.com/wsgdrfz/SIE-CNN/tree/master/1-si-convnet).

 

## Caffe Architecture

Caffe (Convolutional Architexture for Fast Feature Embedding) is a convolutional neural network framework based on C++/CUDA/Python. Basically it can be divided into three architectures, Blobs, Layers and Networks. Everything related to data is through blob, including saving, communication and operation. Layer is the fundermental of model and computation. Network is resposible for integrating and connecting layers.

### Blobs

Blob is the basic data structural unit of Caffe and it is actually a four dimensional array designed to stores and communicates data. It provides a unified memory interface to hold batches of images, model parameters and the derivatives for optimization generated in backpropagtion. There are two classes of data: `data` and `diff`. `data` is the normal data in network while `diff` is the derivatives of network. Blobs can synchronize these data between CPU and GPU in a way ignoring low-level details while maintaining a high level of performance. 

### Layers

Layers are the key of Caffe and there are many computational operations in layers:

1. Convolve filters
2. Pool
3. Take inner products
4. Apply nonlinearities like ReLu, sigmoid
5. Normalize
6. Load data
7. Compute loss

Every layer takes one or more blobs as input and output after performing calculations. 

### Networks

The network in Caffe is a directed acyclic graph of connected layers and a typical network starts from a data layer and ends with a loss layer. Caffe does all the bookkeeping through to ensure the correctness of forward and backward propagation. 



## Main Changes to Caffe

1. `caffe.proto`: declare parameters in messages.

2. `layer_factory.cpp`: initiate and register the new layer.

3. Layer header: define new layers in `/include/caffe/`.

4. `util/transformation.(gpp/cpp/cu)`: add transformation funtions.

5. defining layers:

   ![Scale-invariant Convolution Layer](/Users/wsgdrfz/Library/Application Support/typora-user-images/image-20191120170602274.png)

   - `up_layer.cpp`: Implement the `UpsamplingLayer` which scales only one bottom blob with the functions in `util/transformation.(gpp/cpp/cu)` and outputs layers whose size is same as transformations defined.
   - `tiedconv_layer.cpp`: Implement the convolution part.
   - `downpool_layer.cpp`: Implement the `DownpoolLayer` and it is almost the same as `UpsamplingLayer`. It also standardises the input shape and does the max-pool over scales.
   - `ticonv_layer.cpp`: Wrap `UpsamplingLayer`, `tiedconv_layer` and `DownpoolLayer` in `TIConvolutionLayer`. Note that one `TIConvolutionLayer` is corrsponding to one traditional convolutional layer.

6. related test files in `src/test/` to ensure the correctness of new layers.



## Evaluation

### Evaluation on MNIST-Scale

In this paper, the network is trained and tested on 10,000 and 50,000 images respectively. They evaluate the model on six train/test folds and report the average test error. The network architecture is shown below in protobuff format.

```
name: "MNIST-sicnn-Table-1-split-1"
layers {
   name: "mnist"
   type: HDF5_DATA
   top: "data"
   top: "label"
   hdf5_data_param {
    source: "../../data/mnist/table1/10k_split1_test.txt"
    batch_size: 128
   }
  include: { phase: TRAIN }
}
layers {
   name: "mnist"
   type: HDF5_DATA
   top: "data"
   top: "label"
   hdf5_data_param {
    source: "../../data/mnist/table1/10k_split1_train.txt"
    batch_size: 100
   }
  include: { phase: TEST }
}

layers {
  name: "conv1"
  type: TICONV
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
  transformations {}
  transformations { scale: 0.63 }
  transformations { scale: 0.7937 }
  transformations { scale: 1.2599 }
  transformations { scale: 1.5874 }
  transformations { scale: 2 }
}
layers {
  name: "relu1"
  type: RELU
  bottom: "conv1"
  top: "conv1"
}
layers {
  name: "pool1"
  type: POOLING
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    kernel_size: 2
    stride: 2
    pool: MAX
  }
}

layers {
  name: "conv2"
  type: TICONV
  bottom: "pool1"
  top: "conv2"
  blobs_lr: 1.
  blobs_lr: 2.
  weight_decay: 1.
  weight_decay: 0.
  convolution_param {
    num_output: 64
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
    }
  }
  transformations {}
  transformations { scale: 0.63 }
  transformations { scale: 0.7937 }
  transformations { scale: 1.2599 }
  transformations { scale: 1.5874 }
  transformations { scale: 2 }
}
layers {
  name: "relu2"
  type: RELU
  bottom: "conv2"
  top: "conv2"
}
layers {
  name: "pool2"
  type: POOLING
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    kernel_size: 3
    stride: 3
    pool: MAX
  }
}

layers {
  name: "ip1"
  type: INNER_PRODUCT
  bottom: "pool2"
  top: "ip1"
  blobs_lr: 1.
  blobs_lr: 2.
  weight_decay: 1.
  weight_decay: 0.
  inner_product_param {
    num_output: 150
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "relu3"
  type: RELU
  bottom: "ip1"
  top: "ip1"
}

layers {
  name: "ip2"
  type: INNER_PRODUCT
  bottom: "ip1"
  top: "ip2"
  blobs_lr: 1.
  blobs_lr: 2.
  weight_decay: 1.
  weight_decay: 0.
  inner_product_param {
    num_output: 10
    weight_filler {
      type: "gaussian"
      std: 0.1
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
name: "accuracy"
type: ACCURACY
bottom: "ip2"
bottom: "label"
top: "accuracy"
include: { phase: TEST }
}
layers {
  name: "loss"
  type: SOFTMAX_LOSS
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```

#### Result

We replicate the result of `10k_split1_train/test` and get the test error of 3.28%. 

### Evaluation on Oral Cancer dataset

1. Convert dataset to lmdb/leveldb format

   Firstly, the dataset needs to be converted to lmdb/leveldb format to meet Caffe requirements. This format not only improves the IO efficiency but also acclerates the speed of loading data in training and test. We use the `convert_imageset.cpp` offered by Caffe. In order to use this, we creat the `LISTFILE`  in a way follows the required format. Below is the shell code to generate the `LISTFILE`.

   ```shell
   cancer_file=test_cancer.txt
   healthy_file=test_healthy.txt
   h_path=Healthy
   c_path=Cancer
   
   find $h_path -name *.jpg > $healthy_file
   find $c_path -name *.jpg > $cancer_file
   
   
   sed 's/$/& 1/g' $cancer_file > 'test.txt'
   sed 's/$/& 0/g' $cancer_file >> 'test.txt'
   
   rm $cancer_file
   rm $healthy_file
   ```

2. Train network architecture 

3. Test network architecture

4. Test error with respect to different training size

## Reference

[1] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., Guadarrama, S. and Darrell, T., 2014, November. Caffe: Convolutional architecture for fast feature embedding. In *Proceedings of the 22nd ACM international conference on Multimedia* (pp. 675-678). ACM.