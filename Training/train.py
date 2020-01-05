
import fnmatch
import os

import cv2
import numpy as np

import config
import face


MEAN_FILE = 'mean.png'
POSITIVE_EIGENFACE_FILE = 'positive_eigenface.png'



def walk_files(directory, match='*'):
	
	for root, dirs, files in os.walk(directory):
		for filename in fnmatch.filter(files, match):
			yield os.path.join(root, filename)

def prepare_image(filename):
	
	return face.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def normalize(X, low, high, dtype=None):
	
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	# normalize to [0...1].
	X = X - float(minX)
	X = X / float((maxX - minX))
	# scale to [low...high].
	X = X * (high-low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype=dtype)

if __name__ == '__main__':
	print "Reading training images..."
	faces = []
	labels = []
	pos_count = 0
	neg_count = 0
	# Read all positive images
	for filename in walk_files(config.POSITIVE_DIR, '*.png'):
		faces.append(prepare_image(filename))
		labels.append(config.POSITIVE_LABEL)
		pos_count += 1
	
	print 'Read', pos_count, 'positive images'

	# Train model
	print 'Training model...'
	model = cv2.createEigenFaceRecognizer()
	model.train(np.asarray(faces), np.asarray(labels))

	# Save model results
	model.save(config.TRAINING_FILE)
	print 'Training data saved to', config.TRAINING_FILE

	# Save mean and eignface images which summarize the face recognition model.
	mean = model.getMat("mean").reshape(faces[0].shape)
	cv2.imwrite(MEAN_FILE, normalize(mean, 0, 255, dtype=np.uint8))
	eigenvectors = model.getMat("eigenvectors")
	pos_eigenvector = eigenvectors[:,0].reshape(faces[0].shape)
	cv2.imwrite(POSITIVE_EIGENFACE_FILE, normalize(pos_eigenvector, 0, 255, dtype=np.uint8))
	
