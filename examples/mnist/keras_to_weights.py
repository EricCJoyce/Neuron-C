import sys
import struct
import keras
from keras.models import load_model

'''
python keras_to_weights.py mnist_01.h5
'''

def main():
	conv2dctr = 0
	densectr = 0
	model = load_model(sys.argv[1])

	for i in range(0, len(model.layers)):

		if isinstance(model.layers[i], keras.layers.Conv2D):
			weights = conv2d_to_weights(model.layers[i])				#  Each in 'weights' is a filter

			for weightArr in weights:
				fh = open('Conv2D-' + str(conv2dctr) + '.weights', 'wb')
				packstr = '<' + 'd'*len(weightArr)
				fh.write(struct.pack(packstr, *weightArr))
				fh.close()

				conv2dctr += 1

		elif isinstance(model.layers[i], keras.layers.Dense):
			weights = dense_to_weights(model.layers[i])

			fh = open('Dense-' + str(densectr) + '.weights', 'wb')
			packstr = '<' + 'd'*len(weights)
			fh.write(struct.pack(packstr, *weights))
			fh.close()

			densectr += 1
	return

#  Write layer weights to file in ROW-MAJOR ORDER so our C program can read them into the model
def dense_to_weights(layer):
	ret = []

	w = layer.get_weights()
	width = len(w[1])													#  Number of units
	height = len(w[0])													#  Number of inputs (excl. bias)

	for hctr in range(0, height):										#  This is the row-major read
		for wctr in range(0, width):
			ret.append(w[0][hctr][wctr])

	for wctr in range(0, width):
		ret.append(w[1][wctr])

	return ret

#  Return a list of lists of weights.
#  Each can be written to a buffer and passed as weights to a C Conv2DLayer.
def conv2d_to_weights(layer):
	ret = []

	w = layer.get_weights()
	filterW = len(w[0][0])
	filterH = len(w[0])
	numFilters = len(w[1])

	for fctr in range(0, numFilters):
		ret.append( [] )
		for hctr in range(0, filterH):
			for wctr in range(0, filterW):
				ret[-1].append( w[0][hctr][wctr][0][fctr])
		ret[-1].append(w[1][fctr])

	return ret

if __name__ == '__main__':
	main()
