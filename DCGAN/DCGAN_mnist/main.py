# Implementation of DCGAN - mnist example 
# Tensorflow 1.x 
# Python 3.x 

import tensorflow as tf 
import numpy as np 
import argparse 
import matplotlib 
from string import ascii_lowercase
import random
from tensorflow.examples.tutorials.mnist import input_data
#from helper import Helper as H 

'''
*Hyperparameters*
Mini-batch SGD with mini-batch size of 128
All weights: initialized from zero-centered Normal Dist. with stddev 0.02
LeakyReLU: slope: 0.2
Adam Optimizer 
Learning rate = 0.0002
momentum term = 0.5 
'''

# arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--mnist_data_path", type=str, default="", help="Path to the MNIST data")
parser.add_argument("--img_save_path", type=str, default="", help="Path to where you want to store generated images")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size in integer. Default is 128")
parser.add_argument("--epochs", type=int, default=100, help="Epochs in integer. Default is 100")
parser.add_argument("--mode", type=int, default="generate", choices=["train", "generate"])
args = parser.parse_args()

if args.img_save_path[-1] != "/":
	args.img_save_path += "/"

class DCGAN(object):

	def __init__(self):
		# Saving checkpoint file
		self.model_save_path = args.img_save_path + "model.ckpt"

		# Z: random vector 
		# X: input from MNIST
		# Y: labels - real or fake
		self.Z = tf.placeholder(tf.float32, shape=[None, 100], name="Z")
		self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
		self.Y = tf.placeholder(tf.float32, shape=[None, 2], name="Y")

		self.G = self.generator(self.Z, reuse=False)
		self.D, self.D_logits = self.discriminator(self.X, reuse=False)
		self.D_, self.D_logits_ = self.discriminator(self.G, trainable=False, reuse=True)
		
		self.losses()
		self.initialization()

	def initialization():
		# Save ops for checkpoints
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		try:
			tf.global_variables_initializer().run()
		except:
			tf.initialize_all_variables().run()
		self.sess.run(self.init)

	def generator(self, Z, reuse=True):
		'''
		fractional-strided convolutions 
		use batch normalization except in the output layer
		activation function: ReLU, except for output use Tanh
		No fully connected hidden layers 
		'''
		#Xavier initialization 
		xav_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
		with tf.variable_scope("generator", initializer = xav_init, reuse=reuse, dtype=tf.float32):
			with tf.variable_scope("reshape"):
				G_layer = tf.layers.dense(Z, 7 * 7 * 256, activation=None)
				G_layer = tf.reshape(G_layer, [-1, 7, 7, 256])
				G_layer = tf.layers.batch_normalization(out)
				G_layer = tf.nn.relu(G_layer)
				# result is 7 * 7 * 256

			with tf.variable_scope("Trans Conv Layer 1"):
				G_layer = tf.layers.conv2d_transpose(G_layer, 128, [3, 3], strides=[2,2], padding="same")
				G_layer = tf.layers.batch_normalization(G_layer)
				G_layer = tf.nn.relu(G_layer)
				# result is 14 * 14 * 128
			
			with tf.variable_scope("Trans Conv Layer 2"):
				G_layer = tf.layers.conv2d_transpose(G_layer, 64, [3, 3], strides=[2,2], padding="same")
				G_layer = tf.layers.batch_normalization(G_layer)
				G_layer = tf.nn.relu(G_layer)
				# result is 28 * 28 * 64

			with tf.variable_scope("Trans Conv Layer 3"): # no batch normalization
				G_layer = tf.layers.conv2d_transpose(G_layer, 1, [5, 5], strides=[1,1], padding="same")
				logits = G_layer 
				output = tf.nn.tanh(G_layer)
	return output

	def discriminator(self, X, reuse=True, trainable=True):
		'''
		strided convolutions 
		batch normalization except in input layer 
		activation function: LeakyReLU
		last Conv layer is flattened and fed into a single sigmoid output
		No fully connected hidden layers 
		1st conv layer filter #: 64
	'''
	#Xavier initialization 
		xav_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
		with tf.variable_scope("discriminator", initializer = xav_init, reuse=reuse, dtype=tf.float32):
			# 3 Convolutional Layers
			   	# input layer is 28 * 28 * 1 (MNIST)
			D_layer = conv_layer(X, 64, 5, 2, True, False) # No normalization for the 1st layer 
			#results in 14 * 14 * 64 

			D_layer = conv_layer(D_layer, 128, 3, 2, True, True)
			#results in 7  * 7 * 128

			D_layer = conv_layer(D_layer, 256, 3, 1, True, True)
			#results in 7 * 7 * 256

			flat = tf.reshape(D_layer, [-1, 7*7*256])
			logits = tf.layers.dense(flat, 2, activation=None, traiable=trainable)
			output = tf.sigmoid(logits)
	return output, logits 

	def conv_layer(prev_layer, filters, kernel_size, stride, is_training, batch_norm = True):
		conv_layer = tf.layers.conv2d(prev_layer, filters, [kernel_size, kernel_size], strides=[stride, stride], padding="same", trainable=trainable)
		if batch_norm:
			conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
		conv_layer = tf.nn.leaky_relu(conv_layer, alpha=0.2)
		return conv_layer


	def losses(self):
		'''
		Adam Optimizer 
		learning_rate = 0.0002
		'''
		self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
		self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
		self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D_)))
		self.d_loss = self.d_loss_real + self.d_loss_fake
		
		self.d_train = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="d"))
		self.g_train = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="g"))


	#Train model 
	def train(self):
		# mnist data
		mnist = input_data.read_data_sets(args.mnist_data_path, one_hot=True)

		batch_size = args.batch_size
		epochs = args.epochs
		train_dataset_size = 60000 #MNIST size
		runs_per_epoch = int(train_dataset_size / batch_size)

		# lables: [0,1]
		labels_fake = np.zeros((batch_size, 2), dtype=np.float32)
		labels_fake[:, 1] = 1.0

		# labels: [1,0]
		labels_real = np.zeros((batch_size, 2), dtype=np.float32)
		labels_real[:, 0] = 1.0

		for i in range(epochs):
			for run in range(runs_per_epoch):
				#MNIST training images 
				train_images, test_images = mnist.train.next_batch(batch_size)
				train_images = train_images.reshape((-1, 28, 28, 1)) * 2.0 - 1.0

				#Training Discrimator on real images 
				d_r_train, d_r_cost, debug_y_d_r, debug_d_r = self.session.run([self.d_train, self.d_loss, self.Y, self.D], feed_dict={self.X: train_images, self.Y=labels_real})

				#Generate G(z) - random noise
				z = np.random.standard_normal((batch_size, 100)).astype(np.float32)
				gen_images = self.session.run(self.G, feed_dict={self.Z: z})

				#Training Discriminator on fake images 
				d_f_train, d_f_cost, debug_y_d_f, debug_d_f = self.session.run([self.d_train, self.d_loss, self.Y, self.D], feed_dict={self.X: gen_images, self.Y: labels_fake})

				#Training Generator
				g_train, g_cost, debug_y_g, debug_g, debug_dg = self.session.run([self.g_train, self.g_loss, self.Y, self.G, self.D_], feed_dict={self.Z: z, self.Y: labels_real})

				if run% 50 == 0: 
					print()
					print("[epoch {:>5} | run {:>5}] {:>20} : {:>10.4f}".format(i, run, "D real", d_r_cost))
					print("[epoch {:>5} | run {:>5}] {:>20} : {:>10.4f}".format(i, run, "D fake", d_f_cost))
					print("[epoch {:>5} | run {:>5}] {:>20} : {:>10.4f}".format(i, run, "G", g_cost))
			
			print()
			print("Saving model to {}".format(self.model_save_path))
			self.saver.save(self.session, self.model_save_path)
			print("Finished")

	#Generate images
	def test(self): 
		self.saver.restore(self.session, self.model_save_path)
		z = np.random.standard_normal((args.batch_size, 100)).astype(np.float32)
		gen_imags = self.session.run(self.G, feed_dict={self.Z: z })

		plot_title = "Model's Generated Image"
		file_name = "img_generated_{}.jpg".format(get_random_string(5))
		print()
		print("Image Destination: {}".format(args.img_save_path + fname))
		

	def get_random_string(length):
		chars = ascii_lowercase + "0123456789"
		return "".join(random.choices(chars, k=length))


if __name__ == "__main__":
	dcgan = DCGAN()
	if args.mode == "train":
		dcgan.train()
	elif args.mode == "test":
		dcgan.test()
	else:
		print("Invalid input!")
	print("\nEverything is done")




