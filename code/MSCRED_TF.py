import numpy as np
import tensorflow as tf
import argparse
#import pandas as pd
import os, sys
import timeit
import math
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

parser = argparse.ArgumentParser(description = 'MSCRED encoder-decoder')
parser.add_argument('--batch', type = int, default = 1,
				   help = 'batch_size for data input') #current version supprts batch_size = 1
parser.add_argument('--sensor_n', type = int, default = 30,
				   help = 'number of sensors')
parser.add_argument('--scale_n', type = int, default = 3,
				   help = 'number of matrix scale')
parser.add_argument('--step_max', type = int, default = 5,
				   help = 'maximum step of ConvLSTM')
parser.add_argument('--learning_rate',  type=float, default= 0.0002,
						help='learning rate')
parser.add_argument('--training_iters',  type = int, default = 5,
						help = 'number of maximum training iterations')
parser.add_argument('--train_start_id',  type = int, default = 12,
						help = 'training start id')
parser.add_argument('--train_end_id',  type = int, default = 800,
						help = 'training end id')
parser.add_argument('--test_start_id',  type = int, default = 800,
						help = 'test start id')
parser.add_argument('--test_end_id',  type = int, default = 2000,
						help = 'test end id')
parser.add_argument('--save_model_step', type = int, default = 1,
						help = 'number of iterations to save model')
parser.add_argument('--model_path', type = str, default = '../MSCRED/',
				   help='path to save models')
parser.add_argument('--matrix_data_path', type = str, default = '../data/matrix_data/',
				   help='matrix data path')
# parser.add_argument('--test_input_path', type = str, default = '../data/matrix_data/test_data/',
# 				   help='test input data path')
parser.add_argument('--train_test_label', type=int, default = 1,
				   help='train/test label: train (1), test (0)')
parser.add_argument('--GPU_id', type=int, default = 3,
				   help='GPU ID to select')

args = parser.parse_args()
print(args)

# argument settings
batch_size = args.batch
learning_rate =args.learning_rate
training_iters = args.training_iters
sensor_n = args.sensor_n
scale_n = args.scale_n
step_max= args.step_max

train_start_id = args.train_start_id
train_end_id = args.train_end_id
test_start_id = args.test_start_id
test_end_id = args.test_end_id

model_path = args.model_path
matrix_data_path = args.matrix_data_path
train_data_path = matrix_data_path + "train_data/"
test_data_path = matrix_data_path + "test_data/"
save_model_step = args.save_model_step

train_test_label = args.train_test_label

# set GPU
#GPU_id = args.GPU_id
#os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)

# 4d data
data_input = tf.placeholder(tf.float32, [step_max, sensor_n, sensor_n, scale_n])

# parameters: adding bias weight get similar performance
conv1_W = tf.Variable(tf.zeros([3, 3, scale_n, 32]), name = "conv1_W")
conv1_W = tf.get_variable("conv1_W", shape = [3, 3, scale_n, 32], initializer=tf.contrib.layers.xavier_initializer())
conv2_W = tf.Variable(tf.zeros([3, 3, 32, 64]), name = "conv2_W")
conv2_W = tf.get_variable("conv2_W", shape = [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
conv3_W = tf.Variable(tf.zeros([2, 2, 64, 128]), name = "conv3_W")
conv3_W = tf.get_variable("conv3_W", shape = [2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
conv4_W = tf.Variable(tf.zeros([2, 2, 128, 256]), name = "conv4_W")
conv4_W = tf.get_variable("conv4_W", shape = [2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
# conv5_W = tf.Variable(tf.zeros([2, 2, 256, 256]), name = "conv5_W")
# conv5_W = tf.get_variable("conv5_W", shape = [2, 2, 256, 256], initializer=tf.contrib.layers.xavier_initializer())

# this version computes attention weight based on inner-product between feature representation of last step and other steps
# thus no attenion parameter
# atten_2_W = tf.Variable(tf.random_normal([64, 32]))
# atten_3_W = tf.Variable(tf.random_normal([128, 64]))
# atten_4_W = tf.Variable(tf.random_normal([256, 128]))
# atten_5_W = tf.Variable(tf.random_normal([256, 128]))

# deconv5_W = tf.Variable(tf.zeros([2, 2, 256, 256]), name = "deconv5_W")
# deconv5_W = tf.get_variable("deconv5_W", shape = [2, 2, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
deconv4_W = tf.Variable(tf.zeros([2, 2, 128, 256]), name = "deconv4_W")
deconv4_W = tf.get_variable("deconv4_W", shape = [2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
deconv3_W = tf.Variable(tf.zeros([2, 2, 64, 256]), name = "deconv3_W")
deconv3_W = tf.get_variable("deconv3_W", shape = [2, 2, 64, 256], initializer=tf.contrib.layers.xavier_initializer())
deconv2_W = tf.Variable(tf.zeros([3, 3, 32, 128]), name = "deconv2_W")
deconv2_W = tf.get_variable("deconv2_W", shape = [3, 3, 32, 128], initializer=tf.contrib.layers.xavier_initializer())
deconv1_W = tf.Variable(tf.zeros([3, 3, 3, 64]), name = "deconv1_W")
deconv1_W = tf.get_variable("deconv1_W", shape = [3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer())


def cnn_encoder(input_matrix):
	conv1 = tf.nn.conv2d(
	  input = input_matrix,
	  filter = conv1_W,
	  strides=(1, 1, 1, 1),
	  padding = "SAME")
	conv1 = tf.nn.selu(conv1)

	conv2 = tf.nn.conv2d(
	  input = conv1,
	  filter = conv2_W,
	  strides=(1, 2, 2, 1),
	  padding = "SAME")
	conv2 = tf.nn.selu(conv2)

	conv3 = tf.nn.conv2d(
	  input = conv2,
	  filter = conv3_W,
	  strides=(1, 2, 2, 1),
	  padding = "SAME")
	conv3 = tf.nn.selu(conv3)

	conv4 = tf.nn.conv2d(
	  input = conv3,
	  filter = conv4_W,
	  strides=(1, 2, 2, 1),
	  padding = "SAME")
	conv4 = tf.nn.selu(conv4)

	# conv5 = tf.nn.conv2d(
	#   input = conv4,
	#   filter = conv5_W,
	#   strides=(1, 2, 2, 1),
	#   padding = "SAME")
	# conv5 = tf.nn.selu(conv5)

	#print conv5.get_shape()

	return  conv1, conv2, conv3, conv4


def conv1_lstm(conv1_out):
	convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
				conv_ndims = 2,
				input_shape = [sensor_n, sensor_n, 32],
				output_channels = 32,
				kernel_shape = [2, 2],
				use_bias = True,
				skip_connection = False,
				forget_bias = 1.0,
				initializers = None,
				name="conv1_lstm_cell")

	outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv1_out, time_major = False, dtype = conv1_out.dtype)

	# attention based on transformation of feature representation of last step and other steps
	# outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
	# outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_2_W))
	# outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[32, 1]))/5.0
	# #outputs_mean_W = tf.matmul(outputs_mean_W, atten_2_V)/5.0
	# attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

	# attention based on inner-product between feature representation of last step and other steps
	attention_w = []
	for k in range(step_max):
	  attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
	attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

	outputs = tf.reshape(outputs[0], [step_max, -1])
	outputs = tf.matmul(attention_w, outputs)
	outputs = tf.reshape(outputs, [1, sensor_n, sensor_n, 32])

	return outputs, state[0], attention_w


def conv2_lstm(conv2_out):
	convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
				conv_ndims = 2,
				input_shape = [int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64],
				output_channels = 64,
				kernel_shape = [2, 2],
				use_bias = True,
				skip_connection = False,
				forget_bias = 1.0,
				initializers = None,
				name="conv2_lstm_cell")

	outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv2_out, time_major = False, dtype = conv2_out.dtype)

	# attention based on transformation of feature representation of last step and other steps
	# outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
	# outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_2_W))
	# outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[32, 1]))/5.0
	# #outputs_mean_W = tf.matmul(outputs_mean_W, atten_2_V)/5.0
	# attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

	# attention based on inner-product between feature representation of last step and other steps
	attention_w = []
	for k in range(step_max):
	  attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
	attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

	outputs = tf.reshape(outputs[0], [step_max, -1])
	outputs = tf.matmul(attention_w, outputs)
	outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64])

	return outputs, state[0], attention_w


def conv3_lstm(conv3_out):
	convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
				conv_ndims = 2,
				input_shape = [int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128],
				output_channels = 128,
				kernel_shape = [2, 2],
				use_bias = True,
				skip_connection = False,
				forget_bias = 1.0,
				initializers = None,
				name="conv3_lstm_cell")

	outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv3_out, time_major = False, dtype = conv3_out.dtype)

	# outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
	# outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_3_W))
	# outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[64, 1]))/5.0
	# #outputs_mean_W = tf.matmul(outputs_mean_W, atten_3_V)/5.0
	# attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

	attention_w = []
	for k in range(step_max):
	  attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
	attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

	outputs = tf.reshape(outputs[0], [step_max, -1])
	outputs = tf.matmul(attention_w, outputs)
	outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128])

	return outputs, state[0], attention_w


def conv4_lstm(conv4_out):
	convlstm_layer = tf.contrib.rnn.ConvLSTMCell(
				conv_ndims = 2,
				input_shape = [int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256],
				output_channels = 256,
				kernel_shape = [2, 2],
				use_bias = True,
				skip_connection = False,
				forget_bias = 1.0,
				initializers = None,
				name="conv4_lstm_cell")

	#initial_state = convlstm_layer.zero_state(batch_size, dtype = tf.float32)
	outputs, state = tf.nn.dynamic_rnn(convlstm_layer, conv4_out, time_major = False, dtype = conv4_out.dtype)

	# outputs_mean = tf.reduce_mean(tf.reduce_mean(outputs[0], axis = 1), axis = 1)
	# outputs_mean_W = tf.tanh(tf.matmul(outputs_mean, atten_4_W))
	# outputs_mean_W = tf.matmul(outputs_mean_W, tf.reshape(outputs_mean_W[-1],[128, 1]))/5.0
	# #outputs_mean_W = tf.matmul(outputs_mean_W, atten_4_V)/5.0
	# attention_w = tf.transpose(tf.nn.softmax(outputs_mean_W, dim=0))

	attention_w = []
	for k in range(step_max):
	  attention_w.append(tf.reduce_sum(tf.multiply(outputs[0][k], outputs[0][-1]))/step_max)
	attention_w = tf.reshape(tf.nn.softmax(tf.stack(attention_w)), [1, step_max])

	outputs = tf.reshape(outputs[0], [step_max, -1])
	outputs = tf.matmul(attention_w, outputs)
	outputs = tf.reshape(outputs, [1, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256])

	return outputs, state[0], attention_w


def cnn_decoder(conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
	conv1_lstm_out = tf.reshape(conv1_lstm_out, [1, sensor_n, sensor_n, 32])
	conv2_lstm_out = tf.reshape(conv2_lstm_out, [1, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64])
	conv3_lstm_out = tf.reshape(conv3_lstm_out, [1, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128])
	conv4_lstm_out = tf.reshape(conv4_lstm_out, [1, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256])
	#conv5_lstm_out = tf.reshape(conv5_lstm_out, [1, int(math.ceil(float(sensor_n)/16)), int(math.ceil(float(sensor_n)/16)), 256])

	# deconv5 = tf.nn.conv2d_transpose(
	#   value = conv5_lstm_out,
	#   filter = deconv5_W,
	#   output_shape = [1, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256],
	#   strides = (1, 2, 2, 1),
	#   padding = "SAME")
	# deconv5 = tf.nn.selu(deconv5)
	# deconv5_concat = tf.concat([deconv5, conv4_lstm_out], axis = 3)

	deconv4 = tf.nn.conv2d_transpose(
	  value = conv4_lstm_out,
	  filter = deconv4_W,
	  output_shape = [1, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128],
	  strides = (1, 2, 2, 1),
	  padding = "SAME")
	deconv4 = tf.nn.selu(deconv4)
	deconv4_concat = tf.concat([deconv4, conv3_lstm_out], axis = 3)

	deconv3 = tf.nn.conv2d_transpose(
	  value = deconv4_concat,
	  filter = deconv3_W,
	  output_shape = [1, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64],
	  strides = (1, 2, 2, 1),
	  padding = "SAME")
	deconv3 = tf.nn.selu(deconv3)
	deconv3_concat = tf.concat([deconv3, conv2_lstm_out], axis = 3)

	deconv2 = tf.nn.conv2d_transpose(
	  value = deconv3_concat,
	  filter = deconv2_W,
	  output_shape = [1, sensor_n, sensor_n, 32],
	  strides = (1, 2, 2, 1),
	  padding = "SAME")
	deconv2 = tf.nn.selu(deconv2)

	deconv2_concat = tf.concat([deconv2, conv1_lstm_out], axis = 3)

	deconv1 = tf.nn.conv2d_transpose(
	  value = deconv2_concat,
	  filter = deconv1_W,
	  output_shape = [1, sensor_n, sensor_n, 3],
	  strides = (1, 1, 1, 1),
	  padding = "SAME")
	deconv1 = tf.nn.selu(deconv1)
	deconv1 = tf.reshape(deconv1, [1, sensor_n, sensor_n, 3])
	return deconv1


conv1_out, conv2_out, conv3_out, conv4_out = cnn_encoder(data_input)
conv1_out = tf.reshape(conv1_out, [-1, step_max, sensor_n, sensor_n, 32])
conv2_out = tf.reshape(conv2_out, [-1, step_max, int(math.ceil(float(sensor_n)/2)), int(math.ceil(float(sensor_n)/2)), 64])
conv3_out = tf.reshape(conv3_out, [-1, step_max, int(math.ceil(float(sensor_n)/4)), int(math.ceil(float(sensor_n)/4)), 128])
conv4_out = tf.reshape(conv4_out, [-1, step_max, int(math.ceil(float(sensor_n)/8)), int(math.ceil(float(sensor_n)/8)), 256])

conv1_lstm_attention_out, conv1_lstm_last_out, atten_weight_1 = conv1_lstm(conv1_out)
conv2_lstm_attention_out, conv2_lstm_last_out, atten_weight_2 = conv2_lstm(conv2_out)
conv3_lstm_attention_out, conv3_lstm_last_out, atten_weight_3 = conv3_lstm(conv3_out)
conv4_lstm_attention_out, conv4_lstm_last_out, atten_weight_4 = conv4_lstm(conv4_out)

deconv_out = cnn_decoder(conv1_lstm_attention_out, conv2_lstm_attention_out, conv3_lstm_attention_out, conv4_lstm_attention_out)

# loss function: reconstruction error of last step matrix
loss = tf.reduce_mean(tf.square(data_input[-1] - deconv_out))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = 10)


if train_test_label == 1: # model training
	with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)) as sess:
		sess.run(init)
		start = timeit.default_timer()
		for idx in range(training_iters):
			print ("training iteration " + str(idx) + "...")
			for train_id in range(train_start_id, train_end_id):
				matrix_data_path = train_data_path + "train_data_" + str(train_id) + ".npy"
				matrix_gt = np.load(matrix_data_path)
				matrix_gt = np.transpose(matrix_gt, (0, 2, 3, 1))

				#print np.shape(matrix_gt)

				feed_dict = {data_input: np.asarray(matrix_gt)}
				a, loss_value = sess.run([optimizer, loss], feed_dict)

				# if train_id % 50 == 0:
				# 	save_path = saver.save(sess, model_path + str(idx * 15 + train_id/50) + ".ckpt")

			if idx % save_model_step == 0:
				if not os.path.exists(model_path):
					os.makedirs(model_path)
				save_path = saver.save(sess, model_path + str(idx) + ".ckpt")
				print ("mse of last train data: " + str(loss_value))

		stop = timeit.default_timer()

		print(str(training_iters) + " iterations training finish, use time: " + str(stop - start))

if train_test_label == 0: # model test
	#learning_curve = open("../data/learning_curve.csv", "w")

	with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=2, intra_op_parallelism_threads=2)) as sess:
		restore_idx = 4 # setting restore_idx of learned model
		saver.restore(sess, model_path + str(restore_idx) + ".ckpt")

		reconstructed_data_path = matrix_data_path + "reconstructed_data/"
		if not os.path.exists(reconstructed_data_path):
			os.makedirs(reconstructed_data_path)

		print ("model test: generate recontrucuted matrices"+ "...")
		loss_value_ave = 0
		for test_id in range(test_start_id, test_end_id):
			matrix_data_path = test_data_path + 'test_data_' + str(test_id) + ".npy"
			matrix_gt = np.load(matrix_data_path)
			matrix_gt = np.transpose(matrix_gt, (0, 2, 3, 1))

			feed_dict = {data_input: np.asarray(matrix_gt)}
			reconstructed_matrix = sess.run([deconv_out], feed_dict)

			# loss_value = sess.run([loss], feed_dict)
			# loss_value_ave += loss_value[0]
			# print loss_value

			path_temp = os.path.join(reconstructed_data_path, 'reconstructed_data_' + str(test_id) + ".npy")
			np.save(path_temp, np.asarray(reconstructed_matrix))

			# learning_curve.write("%d\t%lf\n"%(i*50, loss_value_ave/789))
			# print loss_value_ave/789
			# print str(i*50) + "\t" + str(loss_value_ave/789)
		print ("reconstructed matrices generation finish.")

	#learning_curve.close()
