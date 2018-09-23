''' 
Generating new kinds of pokemons using GAN 
codes from https://github.com/llSourcell/Pokemon_GAN/blob/master/pokeGAN.py
'''

import os 
import tensorflow as tf 
import numpy as np 
import cv2 # OpenCV
import random
import scipy.misc
from utils import *

slim = tf.contrib.slim # interface to contrib functions, examples, and models 

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000
version = 'newPokemon'
newPoke_path = './' + version

def lrelu(x, n, leak=0.2): #Leaky relu. Now have it as tf.nn.leaky_relu
	return tf.maximum(x, leak * x, name=n)

def process_data():
	current_dir = os.getcwd()
	# parent = os.path.dirname(current_dir)
	pokemon_dir = os.path.join(current_dir, 'data')
	images = []
	for each in os.listdir(pokemon_dir):
		images.append(os.path.join(pokemon_dir, each))
	#print images
	all_images = tf.convert_to_tensor(images, dtype = tf.string)

	images_queue = tf.train.slice_input_producer([all_images]) # produces a slice of each Tensor in tensor_list 

	content = tf.read_file(images_queue[0])
	image = tf.image.decode_jpeg(content, channels=CHANNEL)
	# sess1 = tf.Session()
	# print(sess1.run(image))

	image = tf.image.random_flip_left_right(image)
	image = tf.image.random_brightness(image, max_delta = 0.1)
	image = tf.image.random_contrast(image, lower =0.9, upper = 1.1 ) # preprocessing, Data Augmentataion
	# noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT, WIDTH, CHANNEL], dtype = tf.float32, stddev = 1e-3, name='noise'))
	# print image.get_shape()

	size = [HEIGHT, WIDTH]
	image = tf.image.resize_images(image, size)
	image.set_shape([HEIGHT, WIDTH, CHANNEL])
	# image = image + noise
	# image = tf.transpose(image, perms=[2, 0, 1])
	# print image.get_shape()

	iamge = tf.cast(image, tf.float32)
	image = image / 255.0

	images_batch = tf.train.shuffle_batch([image], batch_size = BATCH_SIZE, num_threads=4, capacity=200 + 3 * BATCH_SIZE, min_after_dequeue = 200) 
	num_images = len(images)

	return images_batch, num_images