# Hand-Written Digit Classification

This demonstrates the real motivation behind Neuron-C: we want to be able to train a deep network in Keras but use what we learn in stand-alone programs without having to launch Python and load TensorFlow.

## 1. Train a Convolutional Network in Keras

**train_mnist.py** loads the MNIST dataset of hand-written digits, builds a model, and trains. 

### Requirements
- Python
- Keras
- TensorFlow
- Numpy
- Matplotlib (optional, for graphing results)

Your results may vary, but my model reached its best incarnation at epoch 6. It has been saved to **mnist_06.h5**.

![Training and Validation Loss](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/loss.png "Loss")

## 2. Export a Model's Weights

(To be continued)

## 3. Reconstruct the Model from Exported Weights

(To be continued)

## 4. Run Your Model in C

(To be continued)

## 5. Compare Results

(To be continued)
