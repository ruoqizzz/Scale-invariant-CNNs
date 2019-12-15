# Weekly Report 06

## Things we did this week

- Oral Presentation Preparation
- Implement standard CNN and evaluate the network. The details are in the SS-CNN-Implementation.pdf

## Our Questions(Same as last week)

- Still confused with *Log-Radial Harmonics: A scale-steerable filter basis*. We don't understand the theory but know how to use these filters.
- We now change the third paper to *Making Convolutional Network Shift-Invariant Again*. But this paper improves the CNN in a general way rather than focusing on Scale-Invariance/Equivariance. We don't know whether it is a good change but we believe there is a huge connection between shift-invariance and scale-invariance/equivariance. Also the new method can improve the result of any scale-invairance/equivariance network.
- For SS-CNN, wee use the training size of 1k and trained for 300 epochs. The result 4.11±0.13 is a little far from the result in paper 1.91±0.04 because our training size is 1k while author use 10K. That is because the Colab can only run continuously at most 12 hours one time, the training size of 10k would cost more than 20 hours. Do we need to keep the same training size as them? 