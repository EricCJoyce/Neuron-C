# Hand-Written Digit Classification

This demonstrates the real motivation behind Neuron-C: we want to be able to train a deep network in Keras but use what we learn in stand-alone programs without having to launch Python and load TensorFlow. At stake is the difference between a 39-millisecond runtime and a 13-second runtime.

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

| Bitmap                                                                                                 | mnist_06.h5                                                                                                                                                                                          | mnist.nn                                                                                                                                                                                             |
| ------------------------------------------------------------------------------------------------------ |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![MNIST 0](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_0.pgm "0") | <ol start="0"><li><b>0.999912</b></li><li>0.000000</li><li>0.000001</li><li>0.000000</li><li>0.000001</li><li>0.000011</li><li>0.000055</li><li>0.000003</li><li>0.000005</li><li>0.000012</li></ol> | <ol start="0"><li><b>0.999912</b></li><li>0.000000</li><li>0.000001</li><li>0.000000</li><li>0.000001</li><li>0.000011</li><li>0.000055</li><li>0.000003</li><li>0.000005</li><li>0.000012</li></ol> |
| ![MNIST 1](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_1.pgm "1") | <ol start="0"><li>0.000000</li><li><b>0.999929</b></li><li>0.000003</li><li>0.000000</li><li>0.000046</li><li>0.000001</li><li>0.000002</li><li>0.000012</li><li>0.000006</li><li>0.000000</li></ol> | <ol start="0"><li>0.000000</li><li><b>0.999929</b></li><li>0.000003</li><li>0.000000</li><li>0.000046</li><li>0.000001</li><li>0.000002</li><li>0.000012</li><li>0.000006</li><li>0.000000</li></ol> |
| ![MNIST 2](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_2.pgm "2") | <ol start="0"><li>0.000000</li><li>0.000000</li><li><b>1.000000</b></li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li><b>1.000000</b></li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li></ol> |
| ![MNIST 3](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_3.pgm "3") | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li><b>0.999988</b></li><li>0.000000</li><li>0.000002</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000010</li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li><b>0.999988</b></li><li>0.000000</li><li>0.000002</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000010</li></ol> |
| ![MNIST 4](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_4.pgm "4") | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li><b>0.999998</b></li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000002</li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li><b>0.999998</b></li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000002</li></ol> |
| ![MNIST 5](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_5.pgm "5") | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000001</li><li>0.000000</li><li><b>0.999997</b></li><li>0.000001</li><li>0.000000</li><li>0.000000</li><li>0.000001</li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000001</li><li>0.000000</li><li><b>0.999997</b></li><li>0.000001</li><li>0.000000</li><li>0.000000</li><li>0.000001</li></ol> |
| ![MNIST 6](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_6.pgm "6") | <ol start="0"><li>0.000004</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000001</li><li>0.000526</li><li><b>0.999456</b></li><li>0.000000</li><li>0.000013</li><li>0.000000</li></ol> | <ol start="0"><li>0.000004</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000001</li><li>0.000526</li><li><b>0.999456</b></li><li>0.000000</li><li>0.000013</li><li>0.000000</li></ol> |
| ![MNIST 7](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_7.pgm "7") | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000001</li><li>0.000002</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li><b>0.999996</b></li><li>0.000000</li><li>0.000002</li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000001</li><li>0.000002</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li><b>0.999996</b></li><li>0.000000</li><li>0.000002</li></ol> |
| ![MNIST 8](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_8.pgm "8") | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000042</li><li>0.000006</li><li>0.000000</li><li>0.000170</li><li>0.000001</li><li>0.000000</li><li><b>0.999766</b></li><li>0.000016</li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000042</li><li>0.000006</li><li>0.000000</li><li>0.000170</li><li>0.000001</li><li>0.000000</li><li><b>0.999766</b></li><li>0.000016</li></ol> |
| ![MNIST 9](https://github.com/EricCJoyce/Neuron-C/blob/master/examples/mnist/samples/sample_9.pgm "9") | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000146</li><li>0.000000</li><li>0.000000</li><li>0.000252</li><li>0.000004</li><li><b>0.999596</b></li></ol> | <ol start="0"><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000000</li><li>0.000146</li><li>0.000000</li><li>0.000000</li><li>0.000252</li><li>0.000004</li><li><b>0.999596</b></li></ol> |

## Conclusion

Take care when modeling your network in Keras and when writing your own code to absorb exported weights. This is not a one-size-fits-all solution. Thoughtfully design your model, and then mindfully translate its weights.
