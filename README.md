# DeepNeuralNetworkForwardRobustness
MATLAB code for testing DNNs robustness through symbolic propagation of abstract concepts

# Training and testing the DNN
We have slightly extended the [MNIST neural network training and testing](https://it.mathworks.com/matlabcentral/fileexchange/73010-mnist-neural-network-training-and-testing), by Johannes Langelaar, modifying ***digit_train.m*** and ***digit_test.m*** introducing the ReLU, LeakyReLU and linear activation functions.

# Forward robustness
The aim of the code is to promote the study of forward robustness propagating *abstract interpretations*, in particular boxes, and enhancing the computation exploiting symbolic computation.

# symbolic computation
The code exploits MATLAB's symbolic features, such as *isAlways*.

# Theory behind the code
The starting point is the paper [Analyzing Deep Neural Networks with Symbolic Propagation: Towards Higher Precision and Faster Verification](https://arxiv.org/abs/1902.09866), that we have extended to work with the LeakyReLU activation function.

# Backward behavior of DNNs
We have just started exploring the possibility of propagating back, symbolically, boxes from the output layer to the input layer.
