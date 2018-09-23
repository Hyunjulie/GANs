'''
Generator and Discriminator
G: FC --> ( Convolution, bias, activation ) repeat
D: (Convolution, activation, bias) repeat
'''

import os 
import tensorflow as tf 
import numpy as np 
import cv2 # OpenCV
import random
import scipy.misc
from utils import *
import processing

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000
version = 'newPokemon'
newPoke_path = './' + version

def generator(input, random_dim, is_train, reuse=False):
	c4, c8, c16, c32, c63 = 512, 256, 128, 63, 32 #Channel Number
	s4 = 4
	output_dim = CHANNEL #RGB image
	with tf.variable_scope('gen') as scope:
		if reuse:
			scope.reuse_variables()
		w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

		#Convolution, bias, activation --> repeat
		conv1 = tf.reshape(flat_conv1, shape[None, s4, s4, c4], name='conv1')
		bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collection=None, scope='bn1')
		act1 = tf.nn.relu(bn1, name='act1')
		
		# 8 * 8 * 256
		conv2 = tf.layers.conv2d_tranpose(act1, c8, kernel_size=[5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
		bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collection=None, scope='bn2')
		act2 = tf.nn.relu(bn2, name='act2')
		
		# 16 * 16 * 128
		conv3 = tf.layers.conv2d_tranpose(act2, c16, kernel_size=[5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
		bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collection=None, scope='bn3')
		act3 = tf.nn.relu(bn3, name='act3')
		
		# 32 * 32 * 64
		conv4 = tf.layers.conv2d_tranpose(act3, c32, kernel_size=[5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
		bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collection=None, scope='bn4')
		act4 = tf.nn.relu(bn4, name='act4')
	
		# 64 * 64 * 32
		conv5 = tf.layers.conv2d_tranpose(act4, c64, kernel_size=[5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv5')
		bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collection=None, scope='bn5')
		act5 = tf.nn.relu(bn5, name='act5')

		# 128 * 128 * 3
		conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv6')
		act6 = tf.nn.tanh(conv6, name='act6')
		return act6

def discriminator(input, is_train, reuse=False):
	c2, c4, c8, c16 = 64, 128, 256, 512
	with tf.variable_scope('dis') as scope:
		if reuse: 
			scope.reuse_variables()

		#Layer 1: Convolution, activation, bias
		conv1 = tf.layers.conv2d(input, c2, kernel_size = [5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv1')
		bn1 = tf.contrib.layers.batch_norm(con1, is_training = is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn1')
		act1 = lrelu(bn1, n='act1')

		#Layer 2: Convolution, activation, bias
		conv2 = tf.layers.conv2d(act1, c4, kernel_size = [5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv2')
		bn2 = tf.contrib.layers.batch_norm(con2, is_training = is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn2')
		act2 = lrelu(bn2, n='act2')

		#Layer 3: Convolution, activation, bias
		conv3 = tf.layers.conv2d(act2, c8, kernel_size = [5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv3')
		bn3 = tf.contrib.layers.batch_norm(con1, is_training = is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn3')
		act3 = lrelu(bn3, n='act3')

		#Layer 4: Convolution, activation, bias
		conv4 = tf.layers.conv2d(act3, c16, kernel_size = [5,5], strides=[2,2], padding="SAME", kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
		bn4 = tf.contrib.layers.batch_norm(conv4, is_training = is_train, epsilon=1e-5, decay=0.9, updates_collections=None, scope='bn4')
		act4 = lrelu(bn4, n='act4')

		# Starting from act4 
		dim = int(np.prod(act4.get_shape()[1:]))
		fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')

		w2 = tf.get_variable('w2', shape=[fc1.shape[-1],1], dtype=tf.float32, initializer = tf.truncated_normal_initializer(stddev=0.02))
		b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		#WGAN : no sigmoid
		logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
		#DCGAN 
		acted_out = tf.nn.sigmoid(logits)
		return logits 
		# for DCGAN: 
		# return acted_out

def train():
	random_dim = 100 
	with tf.variable_scope('input'):
		#real and fake image placeholders
		real_image = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL], name='real_image')
		random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='random_input')
		is_train = tf.placeholder(tf.bool, name='is_train')

	#WGAN 
	fake_image = generator(random_input, random_dim, is_train)

	real_result = discriminator(real_image, is_train)
	fake_result = discriminator(fake_image, is_train, reuse=True)

	# Discriminator optimizer
	d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result) 
	# Generator optimizer 
	g_loss = -tf.reduce_mean(fake_result)


	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'dis' in var.name]
	g_vars = [var for var in t_vars if 'gen' in var.name]
	trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
	trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)

	#clip discriminator weights
	d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]


	batch_size = BATCH_SIZE
	image_batch, samples_num = process_data()

	batch_num = int(samples_num / batch_size)
	total_batch = 0
	sess = tf.Session()
	saver = tf.train.Saver()  #saves and stores variables 
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	#Training in process
	save_path = saver.save(sess, "/tmp/model.ckpt")
	ckpt = tf.train.latest_checkpoint('./model/' + version)
	saver.restore(sess, save_path)
	coord = tf.train.Coordinator() #implements simple mechanism to coordinate the termination of set of threads
	thread = tf.train.start_queue_runners(sess=sess, coord=coord)

	print("Total Training sample num: %d" %samples_num)
	print("\nBatch size: %d, batch num per epoch: %d, epoch num: %d" %(batch_size, batch_num, EPOCH))
	print("Start Training! ")
	for i in range(EPOCH):
		print("Running epcoch {}/{}....!".format(i, EPOCH))
		for j in range(batch_num):
			print(j)
			d_iters = 5
			g_iters = 1

			train_noice = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
			for k in range(d_iters):
				print(k)
				train_image = sess.run(image_batch)

				#WGAN Clip weights
				sess.run(d_clip)

				#Update discriminator 
				_, dLoss = sess.run([trainer_d, d_loss], feed_dict={random_input: train_noise, real_image: train_image, is_train:True})

			# Update generator
			for k in range(g_iters):
				# train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(tf.float32)
				_, gLoss = sess.run([trainer_g, g_loss], feed_dict={random_input: train_noise, is_train:True})

			#print('train: [%d/%d], d_loss: %f, g_loss:%f' %(i, j, dLoss, gLoss))

		#Save checkpoint every 500 epochs
		if i % 500 == 0:
			if not os.path.exists('./model/' + version):
				os.makedir('./model/' + version)
			saver.save(sess, './model/' + version + '/' + str(i))
		
		#Save images every 50 epochs
		if i % 50 == 0: 
			if not os.path.exists(newPoke_path):
				os.makedir(newPoke_path)
			sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
			imgtest = sess.run(fake_img, feed_dict={random_input: sample_noise, is_train:False})
			#imgtest = imgtest * 255.0
			#imgtest.astype(np.uint8)
			save_images(imgtest, [8,8], newPoke_path + '/epoch' + str(i) + '.jpg')

			print("Train: [%d], d_loss: %f, g_loss: %f" % (i, dLoss, gLoss))

	coord.request_stop()
	coord.join(threads)

'''
def test():
	random_dim = 100 
	with tf.variable_scope('input'):
		real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name = 'real_image')
		random_input = tf.placeholder(tf.float32, shape= [None, random_dim], name = 'rand_input')
		is_train = tf.placeholder(tf.bool, name='is_train')

	#WGAN
	fake_image = generator(random_input, random_dim, is_train)
	real_result = discriminator(real_image, is_train)
	fake_result = discriminator(fake_image, is_train, reuse=True)
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	variables_to_restore = slim.get_variables_to_restore(include=['gen'])
	print(variables_to_restore)
	saver = tf.train.Saver(variables_to_restore)
	skpt = tf.train.latest_checkpoint('./model' + version)
	saver.restore(sess, ckpt)
'''

if __name__ == "__main__":
	train()
	#test()













