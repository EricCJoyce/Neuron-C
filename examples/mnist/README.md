# Hand-Written Digit Classification

This demonstrates the real motivation behind Neuron-C: we want to be able to train a deep network in Keras but use what we learn in stand-alone programs without having to launch Python and load TensorFlow.

## 1. Train a Convolutional Network in Keras

`train_mnist.py` loads the MNIST dataset of hand-written digits, builds a model, and trains. 

### Requirements
- Python
- Keras
- TensorFlow
- Numpy
- Matplotlib (optional, for graphing results)

Your results may vary, but my model reached its best incarnation at epoch 6. It has been saved to `mnist_06.h5`.

![Training and Validation Loss](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/loss.png "loss.png")

The important thing to notice in this Python script is how the model is built. I designed the Neuron-C data structures to appeal to my intuition and organizational habits. The price of portability to my library is that I have to build Keras models with translation in mind. Beginning on line 38, this rather unlovely barrage of flattening and lambda-ing ensures that the convolutional filters Keras learned will align with the fully-connected layers that have learned to respond to them. The effect is illustrated below. Keras interleaves filter output; the Neuron-C Accumulator Layer expects them to be grouped.

![Sketch of Filter Alignment](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/filter_alignment.png "filter_alignment.png")

## 2. Export a Model's Weights

`keras_to_weights.py` takes your trained model (or mnist_06.h5 if you'd rather use that), iterates through its layers, and writes each to file.

Weights exported from mnist_06.h5 have been put in the `06` directory.

## 3. Reconstruct the Model from Exported Weights

`build_neuron_model.c` builds layers, populates them with the weights you just exported, links the layers, and writes the network to `mnist.nn`.

## 4. Run Your Model in C

`run.c` expects an `.nn` file and a 28-by-28 `.pgm`. Appropriate `.pgm`s for each of 10 digits can be found in the `samples` folder. `run` loads the bitmap, converts it to floating-point values, feeds it to the network, and prints the network's probability distribution over `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`, reflecting its confidence that the numeral depicted in the bitmap is the corresponding digit.

## 5. Compare Results

`compare.py` is a sanity check. We want to be sure that what the Neuron-C model outputs is the same as what the Keras model outputs.

## Conclusion

Take care when modeling your network in Keras and when writing your own code to absorb exported weights. This is not a one-size-fits-all solution. Thoughtfully design your model, and then mindfully translate its weights.


| Bitmap                                                                                                 | mnist_06.h5   | mnist.nn  |
| ------------------------------------------------------------------------------------------------------ |:-------------:| ---------:|
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_0.pgm "0") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_1.pgm "1") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_2.pgm "2") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_3.pgm "3") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_4.pgm "4") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_5.pgm "5") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_6.pgm "6") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_7.pgm "7") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_8.pgm "8") |               |           |
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_9.pgm "9") |               |           |
