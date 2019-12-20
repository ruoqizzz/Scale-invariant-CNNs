# Weekly Report 04

Date 01/12/2019

## Things we did this week

- This week we completed the implementation of the first paper [Locally Scale-Invariant Convolutional Neural Network](https://arxiv.org/abs/1412.5104). Note that the test error we reported last week is not correct, as each split should be trained for 700 epochs. After more than one hour training, the test error on split 1 we got is 2.93%, which is close to the 2.91% stated on the author's GitHub. We attached one of the 6 outputs in the email.
- We read the second paper [Scale Steerable Filters for Locally Scale-Invariant Convolutional Neural Networks](https://arxiv.org/abs/1906.03861) in detail and wrote a summary of it as PDF file 'SS-CNN-Summary.pdf'.
- We also tried to run the implementation of the second paper from [Github](https://github.com/rghosh92/SS-CNN) on Colab and met some minor problems. We will make it next week.



## Questions

- Can you give us some comments regarding the summaries and the implementation reports? As we will use them in the final report, we hope to get some feedback such that we can revise in time.
- The first two papers both run experiments on MNIST-Scale dataset, and they created the MNIST-Scale with the same method (with a random scaling factor $s\in(0.3,1)$). Are we supposed to create the MNIST-Scale dataset by ourselves and compare the test errors of the two methods on the same dataset, or can we use the existing results?
- The comparison of the three methods on Oral Cancer dataset should include different training sizes. Does the same requirement apply for MNIST-Scale? We have a little disagreements on this...