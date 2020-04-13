'''
  Eric C. Joyce, Stevens Institute of Technology, 2020

  Train a convolutional neural network to classify hand-written digits using Keras and the MNIST data set.
  Notice that if we are to convert weights learned in Keras to an independent C program, then we have to
  control the arrangements of outputs. That is the function of the Lambda layers below. Output from
  convolution must be flattened, and then the flattened output must be re-routed to match the format of
  Neuron-C's Accumulation Layer type.
'''

from keras import models
from keras.layers import Input, Dense, Flatten, Conv2D, Lambda, concatenate, Dropout
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

epochs = 20
batchSize = 128

def main():
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images = train_images.reshape( (60000, 28, 28, 1) )		#  Reshape, clean up
	train_images = train_images.astype('float32') / 255.0
	test_images = test_images.reshape( (10000, 28, 28, 1) )
	test_images = test_images.astype('float32') / 255.0
	train_labels = to_categorical(train_labels)
	test_labels = to_categorical(test_labels)

	imgInput = Input(shape=(28, 28, 1))								#  (height, width, channels)

	#        8 filters,  h, w                                                    h,  w
	conv3x3 = Conv2D(8, (3, 3), activation='relu', padding='valid', input_shape=(28, 28, 1), name='conv3x3')(imgInput)
	#        16 filters,  h, w                                                    h,  w
	conv5x5 = Conv2D(16, (5, 5), activation='relu', padding='valid', input_shape=(28, 28, 1), name='conv5x5')(imgInput)

	flat3x3 = Flatten(name='flat3x3')(conv3x3)						#  length = 8 * (28 - 3 + 1) * (28 - 3 + 1) = 5408
	lambda3x3_0 = Lambda(lambda x: x[:, 0::8])(flat3x3)				#  8 filters: take every 8th from [0]
	lambda3x3_1 = Lambda(lambda x: x[:, 1::8])(flat3x3)				#  8 filters: take every 8th from [1]
	lambda3x3_2 = Lambda(lambda x: x[:, 2::8])(flat3x3)				#  8 filters: take every 8th from [2]
	lambda3x3_3 = Lambda(lambda x: x[:, 3::8])(flat3x3)				#  8 filters: take every 8th from [3]
	lambda3x3_4 = Lambda(lambda x: x[:, 4::8])(flat3x3)				#  8 filters: take every 8th from [4]
	lambda3x3_5 = Lambda(lambda x: x[:, 5::8])(flat3x3)				#  8 filters: take every 8th from [5]
	lambda3x3_6 = Lambda(lambda x: x[:, 6::8])(flat3x3)				#  8 filters: take every 8th from [6]
	lambda3x3_7 = Lambda(lambda x: x[:, 7::8])(flat3x3)				#  8 filters: take every 8th from [7]

	flat5x5 = Flatten(name='flat5x5')(conv5x5)						#  length = 16 * (28 - 5 + 1) * (28 - 5 + 1) = 9216
	lambda5x5_0 = Lambda(lambda x: x[:, 0::16])(flat5x5)			#  16 filters: take every 16th from [0]
	lambda5x5_1 = Lambda(lambda x: x[:, 1::16])(flat5x5)			#  16 filters: take every 16th from [1]
	lambda5x5_2 = Lambda(lambda x: x[:, 2::16])(flat5x5)			#  16 filters: take every 16th from [2]
	lambda5x5_3 = Lambda(lambda x: x[:, 3::16])(flat5x5)			#  16 filters: take every 16th from [3]
	lambda5x5_4 = Lambda(lambda x: x[:, 4::16])(flat5x5)			#  16 filters: take every 16th from [4]
	lambda5x5_5 = Lambda(lambda x: x[:, 5::16])(flat5x5)			#  16 filters: take every 16th from [5]
	lambda5x5_6 = Lambda(lambda x: x[:, 6::16])(flat5x5)			#  16 filters: take every 16th from [6]
	lambda5x5_7 = Lambda(lambda x: x[:, 7::16])(flat5x5)			#  16 filters: take every 16th from [7]
	lambda5x5_8 = Lambda(lambda x: x[:, 8::16])(flat5x5)			#  16 filters: take every 16th from [8]
	lambda5x5_9 = Lambda(lambda x: x[:, 9::16])(flat5x5)			#  16 filters: take every 16th from [9]
	lambda5x5_10 = Lambda(lambda x: x[:, 10::16])(flat5x5)			#  16 filters: take every 16th from [10]
	lambda5x5_11 = Lambda(lambda x: x[:, 11::16])(flat5x5)			#  16 filters: take every 16th from [11]
	lambda5x5_12 = Lambda(lambda x: x[:, 12::16])(flat5x5)			#  16 filters: take every 16th from [12]
	lambda5x5_13 = Lambda(lambda x: x[:, 13::16])(flat5x5)			#  16 filters: take every 16th from [13]
	lambda5x5_14 = Lambda(lambda x: x[:, 14::16])(flat5x5)			#  16 filters: take every 16th from [14]
	lambda5x5_15 = Lambda(lambda x: x[:, 15::16])(flat5x5)			#  16 filters: take every 16th from [15]
																	#  Output length = 14624
	convConcat = concatenate([lambda3x3_0,  lambda3x3_1,  lambda3x3_2,  lambda3x3_3,  \
	                          lambda3x3_4,  lambda3x3_5,  lambda3x3_6,  lambda3x3_7,  \
	                          lambda5x5_0,  lambda5x5_1,  lambda5x5_2,  lambda5x5_3,  \
	                          lambda5x5_4,  lambda5x5_5,  lambda5x5_6,  lambda5x5_7,  \
	                          lambda5x5_8,  lambda5x5_9,  lambda5x5_10, lambda5x5_11, \
	                          lambda5x5_12, lambda5x5_13, lambda5x5_14, lambda5x5_15  ])

	dropout1 = Dropout(0.5)(convConcat)
	dense100 = Dense(100, activation='relu', name='dense400')(dropout1)
	dropout2 = Dropout(0.5)(dense100)
	dense10 = Dense(10, activation='softmax', name='dense10')(dropout2)

	model = models.Model(inputs=imgInput, output=dense10)

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()													#  Print the details

	filepath = 'mnist_{epoch:02d}.h5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacksList = [checkpoint]

	history = model.fit( [train_images], [train_labels], \
	                     epochs=epochs, batch_size=batchSize, callbacks=callbacksList, \
	                     validation_data=[test_images, test_labels] )

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	fig = plt.figure(figsize=(6, 4))

	plt.plot(range(1, epochs + 1), loss, 'bo', label='Train.Loss')
	plt.plot(range(1, epochs + 1), val_loss, 'r', label='Val.Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.title('Training and Validation Loss')
	plt.legend()
	plt.tight_layout()
	plt.show()

	return

if __name__ == '__main__':
	main()
