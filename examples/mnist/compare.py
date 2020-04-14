'''
  Eric C. Joyce, Stevens Institute of Technology, 2020
  
  Finally, let's compare our C-translation against the Keras original.
  e.g.  
   python compare.py mnist_06.h5 mnist.nn samples/sample_1.pgm

  sys.argv[0] = compare.py
  sys.argv[1] = Keras model file name (.h5)
  sys.argv[2] = neuron model file name (.nn)
  sys.argv[3] = PGM filepath
'''

import sys
import subprocess
import numpy as np
from keras.models import load_model

def main():
	kerasFile = sys.argv[1]
	neuronFile = sys.argv[2]
	pgmfilename = sys.argv[3]

	model = load_model(kerasFile)
	img = loadPGM(pgmfilename)
	img = img.reshape((28, 28, 1))

	print('\nThese should be the same:')

	y_hat = model.predict( [[img]] )
	print('Keras model returned:')
	for i in range(0, len(y_hat[0])):
		print("%.6f" % y_hat[0][i])

	args = ['./run', neuronFile, pgmfilename]
	out = subprocess.check_output(args)
	out = out.decode("utf-8").split('\n')[:-1]
	print('\nNeuron-C model returned:')
	for i in range(0, len(out)):
		print(out[i].split()[1])

	return

def loadPGM(filename):
	fh = open(filename, 'rb')

	assert fh.readline() == b'P5\n'

	line = fh.readline()
	while line[0] == ord('#'):
		line = fh.readline()

	width, height = [int(i) for i in line.split()]

	line = fh.readline()
	while line[0] == ord('#'):
		line = fh.readline()

	maxgray = int(line)
	assert maxgray <= 255

	pixels = []
	for y in range(0, width * height):
		pixels.append( float(ord(fh.read(1))) / 255.0)

	fh.close()

	return np.array(pixels)

if __name__ == '__main__':
	main()
