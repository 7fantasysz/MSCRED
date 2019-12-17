import numpy as np
import argparse
import pandas as pd
import os, sys
import math
import scipy
#import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import spatial
import itertools as it
import string
import re

parser = argparse.ArgumentParser(description = 'Signature Matrix Generator')
parser.add_argument('--ts_type', type = str, default = "node",
				   help = 'type of time series: node or link')
parser.add_argument('--step_max', type = int, default = 5,
				   help = 'maximum step in ConvLSTM')
parser.add_argument('--gap_time', type = int, default = 10, # tride width...
				   help = 'gap time between each segment')
parser.add_argument('--win_size', type = int, default = [10, 30, 60],
				   help = 'window size of each segment')
parser.add_argument('--min_time', type = int, default = 0,
				   help = 'minimum time point')
parser.add_argument('--max_time', type = int, default = 20000,
				   help = 'maximum time point')
parser.add_argument('--train_start_point',  type = int, default = 0,
						help = 'train start point')
parser.add_argument('--train_end_point',  type = int, default = 8000,
						help = 'train end point')
parser.add_argument('--test_start_point',  type = int, default = 8000,
						help = 'test start point')
parser.add_argument('--test_end_point',  type = int, default = 20000,
						help = 'test end point')
parser.add_argument('--raw_data_path', type = str, default = '../data/synthetic_data_with_anomaly-s-1.csv',
				   help='path to load raw data')
parser.add_argument('--save_data_path', type = str, default = '../data/',
				   help='path to save data')

args = parser.parse_args()
print(args)

ts_type = args.ts_type
step_max = args.step_max
min_time = args.min_time
max_time = args.max_time
gap_time = args.gap_time
win_size = args.win_size

train_start = args.train_start_point
train_end = args.train_end_point
test_start = args.test_start_point
test_end = args.test_end_point

raw_data_path = args.raw_data_path
link_name_path = "../data/synthetic_data_link_name.csv"
save_data_path = args.save_data_path

value_colnames = ['total_count','error_count','error_rate']
ts_colname="agg_time_interval"
agg_freq='5min'

scale_n = len(win_size) * len(value_colnames)

matrix_data_path = save_data_path + "matrix_data/"
if not os.path.exists(matrix_data_path):
	os.makedirs(matrix_data_path)


def generate_signature_matrix_node():
	data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)
	sensor_n = data.shape[0]
	#data  = np.array(pd.read_csv(raw_data_path, header = None))[:,2:-1]

	# min-max normalization
	max_value = np.max(data, axis=1)
	min_value = np.min(data, axis=1)
	data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)

	# std normalization
	# data = np.nan_to_num(data)
	# data_mean = np.mean(data, axis = 0)
	# data_std = np.std(data, axis = 0)
	# data = np.transpose(data) - data_mean
	# data = data / (data_std + 1e-5)

	data = np.transpose(data)

	# plt.plot(data[3,:])
	# plt.show()

	#multi-scale signature matix generation
	for w in range(len(win_size)):
		matrix_all = []
		win = win_size[w]
		print ("generating signature with window " + str(win) + "...")
		for t in range(min_time, max_time, gap_time):
			#print t
			matrix_t = np.zeros((sensor_n, sensor_n))
			if t >= 60:
				for i in range(sensor_n):
					for j in range(i, sensor_n):
						#if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
						matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
						matrix_t[j][i] = matrix_t[i][j]
			matrix_all.append(matrix_t)

			# if t == 70:
			# 	print matrix_all[6][0]

		path_temp = matrix_data_path + "matrix_win_" + str(win)
		#print np.shape(matrix_all[0])

		np.save(path_temp, matrix_all)
		del matrix_all[:]

	print ("matrix generation finish!")



def link_data_cleaning():
	'''
	colnames = ['source_node_type','destination_node_type',ts_colname, v0, v1, v2, ...]
	'''
	raw_data = pd.read_csv(raw_data_path)

	raw_data.columns = ['source_node_type','destination_node_type',ts_colname] + value_colnames

	# source destination node
	link_name = np.array(raw_data.iloc[:,:2])
	link_name_concat = np.array([x+"-"+y for x,y in link_name])
	unique_link_name = np.unique(link_name)
	unique_link_name_concat = np.unique(link_name_concat)
	sensor_n = unique_link_name.shape[0]

	# std normalization
	data = np.nan_to_num(raw_data.iloc[:,3:])
	data = np.nan_to_num(data)
	data_mean = np.mean(data, axis = 0)
	data_std = np.std(data, axis = 0)
	data = data - np.array([data_mean for i in range(data.shape[0])])
	data = data / (data_std + 1e-5)
	raw_data.iloc[:,3:] = data

	# join to align and pivot
	raw_data['sd_concat'] = link_name_concat
	raw_data[ts_colname] = pd.to_datetime(raw_data[ts_colname],infer_datetime_format=True)
	secon_smallest_ts = raw_data[[ts_colname]].drop_duplicates().nsmallest(2, ts_colname).iloc[1,0]
	date_range = pd.date_range(start=secon_smallest_ts, end=raw_data[ts_colname].max(), freq=agg_freq)
	joined_df = pd.DataFrame(date_range, columns=[ts_colname]).merge(raw_data, on=ts_colname, how='left')


	for value_col in value_colnames:
		pivot_data = joined_df.pivot(index='sd_concat', columns='agg_time_interval', values=value_col)
		output_df = pd.DataFrame(unique_link_name_concat, columns=['sd_concat']).merge(pivot_data.reset_index(),
							on='sd_concat', how='left')
		output_df.to_csv("/home/zhaos/ts_data_csv2/channel_"+value_col+".csv", header=True, index=False)





def generate_signature_matrix_link():
	'''
	For a link a->b,
	sum(vector(a->b) * vectro(b->c) + vector(a->b) * vectro(b->d) + ...)/count(b->c,d,...) +
	sum(vector(a->b) * vectro(z->a) + vector(a->b) * vectro(y->a) + ...)/count(z,y,...->a)
	'''
	for value_col in value_colnames:
		data = pd.read_csv("/home/zhaos/ts_data_csv2/channel_"+value_col+".csv")
		link_name_concat = data.iloc[:,1].values
		data = data.iloc[:,1:].values
		
		#multi-scale signature matix generation
		for w in range(len(win_size)):
			matrix_all = []
			win = win_size[w]
			print ("generating signature with window " + str(win) + "...")
			for t in range(min_time, max_time, gap_time):
				#print t
				matrix_t = np.zeros((sensor_n, sensor_n))
				if t >= 60:
					for i in range(sensor_n): # source node
						for j in range(i, sensor_n): # destination node
							sumprod = 0 # node_i -> node_j
							source_node, dest_node = unique_link_name[i], unique_link_name[j]
							sd_concat_index = np.where(link_name_concat == source_node+"-"+dest_node)[0]
							if len(sd_concat_index) > 0:
								link_vector = data[sd_concat_index[0], t - win:t]
								for k in range(link_name.shape[0]):
									if link_name[k][0] == dest_node:
										# sum(vector(a->b) * vectro(b->c) + vector(a->b) * vectro(b->d) + ...)/count(b->c,d,...)
										sumpprod += np.inner(link_vector, data[k, t - win:t])/(win)
									if link_name[k][1] == source_node:
										# sum(vector(a->b) * vectro(z->a) + vector(a->b) * vectro(y->a) + ...)/count(z,y,...->a)
										sumpprod += np.inner(data[k, t - win:t], link_vector)/(win)
								matrix_t[i][j] = sumpprod
				matrix_all.append(matrix_t)
				
				# if t == 70:
				# 	print matrix_all[6][0]
				
			path_temp = "/home/zhaos/ts_data_csv2/signature_matrix/matrix_win_" + str(win) + str(value_col)
			#print np.shape(matrix_all[0])
			
			np.save(path_temp, matrix_all)
			del matrix_all[:]
			
		print ("matrix generation finish!")




def generate_train_test_data():
	#data sample generation
	print ("generating train/test data samples...")
	matrix_data_path = "/home/zhaos/ts_data_csv2/signature_matrix/"

	train_data_path = matrix_data_path + "train_data/"
	if not os.path.exists(train_data_path):
		os.makedirs(train_data_path)
	test_data_path = matrix_data_path + "test_data/"
	if not os.path.exists(test_data_path):
		os.makedirs(test_data_path)

	data_all = []
	for value_col in value_colnames:
		for w in range(len(win_size)):
			path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + str(value_col) + ".npy"
			data_all.append(np.load(path_temp))

	train_test_time = [[train_start, train_end], [test_start, test_end]]
	for i in range(len(train_test_time)):
		for data_id in range(int(train_test_time[i][0]/gap_time), int(train_test_time[i][1]/gap_time)):
			#print data_id
			step_multi_matrix = []
			for step_id in range(step_max, 0, -1):
				multi_matrix = []
				for k in range(len(value_colnames)):
					for i in range(len(win_size)):
						multi_matrix.append(data_all[k*len(win_size) + i][data_id - step_id])
				step_multi_matrix.append(multi_matrix)

			if data_id >= (train_start/gap_time + win_size[-1]/gap_time + step_max) and data_id < (train_end/gap_time): # remove start points with invalid value
				path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)
			elif data_id >= (test_start/gap_time) and data_id < (test_end/gap_time):
				path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)

			#print np.shape(step_multi_matrix)

			del step_multi_matrix[:]

	print ("train/test data generation finish!")


if __name__ == '__main__':
	'''need one more dimension to manage mulitple "features" for each node or link in each time point,
	this multiple features can be simply added as extra channels
	'''

	if ts_type == "node":
		generate_signature_matrix_node()
	elif ts_type == "link":
		generate_signature_matrix_link()

	generate_train_test_data()
