# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# tqdm
from tqdm import tqdm
# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx
import os
# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler

file_name = 'dataset'
current_dir = os.getcwd()  
output_file_path = os.path.join(current_dir, file_name)  

example_mode = False
if example_mode == True:
	embedding_table_gather_reduce_access = [[0, [2, 4, 0, 2, 4]], [1, [1, 3, 4]], [0, [2, 3, 4, 5]], [1, [1, 2, 4, 2, 5]]]
	offset_global  = [[[0, 2], [0, 1]], [[0, 1], [0, 3]]]

def training_trace_standard(embedding_table_gather_reduce_access, embedding_table_len_global, size_of_the_reduced_embedding_vector_global, offset_global):
	# print("embedding_table_gather_reduce_access",embedding_table_gather_reduce_access)
	# print("embedding_table_len_global",embedding_table_len_global)
	# print("size_of_the_reduced_embedding_vector_global",size_of_the_reduced_embedding_vector_global)
	# print("offset_global",offset_global)
	#print("here")
	total_length = sum(embedding_table_len_global)
	#print("here1")
	# memory_index = list(range(total_length)) # 0 ~ total_length-1
	table_size_list = [size for size in embedding_table_len_global]
	#print("here2")

	#embedding_table_gather_reduce_access = [[elem[0], elem[1].tolist()] for elem in embedding_table_gather_reduce_access] # to list
	embedding_table_gather_reduce_access = [[elem[0], elem[1].tolist()] for elem in embedding_table_gather_reduce_access] # to list
	# print("***embedding_table_gather_reduce_access", embedding_table_gather_reduce_access)
	offset_global = offset_global.unsqueeze(0)
	offset_global = offset_global.tolist()  # to list
	
	#offset_global = [tensor for tensor in offset_global]
	#print("here4")
	
	# print('table_size_list', table_size_list)
	#print ("offset_global",offset_global)
	batch_num = len(offset_global)
	
	#print("here5")
	print('batch_num', batch_num)
	batched_table_access = []
	len_entries = []
	for b in range(batch_num):
		batched_table_access.append([b])
	
	current_batch = 0
	counter = 0
	print("here6")
	for i in range(len(embedding_table_gather_reduce_access)):
			len_entries.append(len(embedding_table_gather_reduce_access[i][1]))
			for j in embedding_table_gather_reduce_access[i][1]:
				batched_table_access[i // len(table_size_list)].append((embedding_table_gather_reduce_access[i][0], j))
		

	#print("len_entries",len_entries)
	print("here7")
	# print('batched_table_access', batched_table_access)
	batched_table_access_list = []
	# Modify the format
	# list_memory2  = sys.getsizeof(batched_table_access)
	# element_memory2 = sum(sys.getsizeof(elem) for elem in batched_table_access)
	# total_memory2 = list_memory2 + element_memory2
	#print("batched_table_access size: ",total_memory2)

	for sublist in batched_table_access:
		new_sublist = [[sublist[0]], sublist[1:]]
		batched_table_access_list.append(new_sublist)
		# if(len(batched_table_access_list)%10000 == 0):
		# 	list_memory  = sys.getsizeof(batched_table_access_list)
		# 	element_memory = sum(sys.getsizeof(ele) for ele in batched_table_access_list)
		# 	total_memory = list_memory + element_memory
			#print("batched_table_access_list size: ",total_memory)
	print("here8")

	# print('batched_table_access_list', batched_table_access_list)
	# batched_table_access_list == [[nth batch], [(kth table, kth entry), (kth table, kth entry), (kth table, kth entry)]]
	# [[[0], [(0, 2), (0, 4), (0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (1, 4)]], [[1], [(2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 2), (3, 5)]]]
	emb_table_pair = []
	emb_table_pair = [(i, j) for i, size in enumerate(table_size_list) for j in range(size)]
	print("here9")
	# print(emb_table_pair)
	# [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
	mapped_dict = {idx: pair for idx, pair in enumerate(emb_table_pair)}
	print("here10")
	# print(mapped_dict)
	# mapped_dict == {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (1, 0), 7: (1, 1), 8: (1, 2), 9: (1, 3), 10: (1, 4), 11: (1, 5)}

	# print(len_entries)
	#len_entries == [5, 3, 4, 5]
	# offset_global == [[[0, 2], [0, 1]], [[0, 1], [0, 3]]]
	entoffset = []
	# print("len(offset_global[0])",len(offset_global[0]))
	# print("len(offset_global[0][0])",len(offset_global[0][0]))
	
	for i in range(len(offset_global)):
		temp_result = []
		for j in range(len(offset_global[i])):
			temp_entry = offset_global[i][j] + [len_entries.pop(0)]
			temp_result.append(temp_entry)
			
		entoffset.append(temp_result)
	
	
	# for j in range(len(offset_global[i])):
	# 	temp_entry = offset_global[i][j] + [len_entries.pop(0)]
	# 	temp_result.append(temp_entry)
		
	# entoffset.append(temp_result)
	print("here11")
	
	# print(entoffset)

	#entoffset == [[[0, 2, 5], [0, 1, 3]], [[0, 1, 4], [0, 3, 5]]]
	# 5 entries wiht offset 0, 2 ......
	# [[0,0,1,1,1,2,3,3], [0,1,1,1,2,2,2,3,3]]
	entry_to_bag = [[] for _ in range(batch_num)]
	print("here12")
	for idx, group in enumerate(entoffset):
		counter = 0
		curr = 1
		for i in range(len(group)):
			for j in range(group[i][-1]):
				# print("i",i)
				# print("j",j)
				# print("curr",curr)
				if (curr < len(group[i])) and (len(group[i]) != 1):
					# print("group[i][curr]",group[i][curr])
					if j >= group[i][curr]:
						counter += 1
						curr += 1
				# print("counter",counter)
				entry_to_bag[idx].append(counter)
			counter += 1
			curr = 1
	print("here13")
	# Classify the lookup according to different bags, ex. [[2,4,0,2,4],[1,3,4]] 2,4=> bag0, 0,2,4=>bag1 1=>bag2 3,4=>bag3
	# res == [[0, 0, 1, 1, 1, 2, 3, 3], [0, 1, 1, 1, 2, 2, 2, 3, 3]] # which entries belongs to which bag(res) (this example is two iteration)
	# print('entry_to_bag: ', entry_to_bag)

	gather_op_access = [[] for _ in range(batch_num)]
	print("here14")
	reverse_mapped_dict = {v: k for k, v in mapped_dict.items()}
	for idx, (_, accesses) in enumerate(batched_table_access_list):
		print ("idx",idx)
		# print("accesses",accesses)
		# print("mapped_dict.items()",mapped_dict.items())
		# print ("len(accesses)",len(accesses))
		# print ("accesses",accesses)
		# print ("len(mapped_dict)",len(mapped_dict))
		# print ("mapped_dict",mapped_dict)
		mapped_indexes = []
		# for access in accesses:
		# 	for k, v in mapped_dict.items():
		# 		if v == access:
		# 			mapped_indexes.append(k)
					
		# 			break
		for access in accesses:
			# mapped_indexes.append(list(mapped_dict.keys())[list(mapped_dict.values()).index(access)])
			mapped_indexes.append(reverse_mapped_dict.get(access))  # O(1) dictionary lookup

		#print("mapped_indexes",mapped_indexes)
		gather_op_access[idx].extend(mapped_indexes)
	print("here15")

	# print('gather_op_access: ', gather_op_access)
	# gather_op_access == [[2, 4, 0, 2, 4, 7, 9, 10], [2, 3, 4, 5, 7, 8, 10, 8, 11]]
	entry_to_bag_extend_to_mem_addr = [[val + len(mapped_dict) for val in sublist] for sublist in entry_to_bag]
	print("here16")
	# print('entry_to_bag_extend_to_mem_addr: ', entry_to_bag_extend_to_mem_addr)
	# entry_to_bag_extend_to_mem_addr == [[12, 12, 13, 13, 13, 14, 15, 15], [12, 13, 13, 13, 14, 14, 14, 15, 15]]
	mem_trace = [[] for _ in range(batch_num)]
	print("here17")
	# for inference gather reduce
	for idx, access in enumerate(gather_op_access):
		for i in range(len(access)):
			mem_trace[idx].append((gather_op_access[idx][i], 'R'))
			mem_trace[idx].append((entry_to_bag_extend_to_mem_addr[idx][i], 'R'))
			mem_trace[idx].append((entry_to_bag_extend_to_mem_addr[idx][i], 'W'))
			
		#print(f"Size of mem_trace0: {sys.getsizeof(mem_trace)/(2**20)} MB")
	print("here18")
	# print('mem_trace: ', mem_trace) # for inference gather reduce ok
	
	# size_of_the_reduced_embedding_vector_global == 4 for example
	gradients_mem_addr = []
	# gradients write back
	for i in range(size_of_the_reduced_embedding_vector_global):
		gradients_mem_addr.append(i + len(mapped_dict) + size_of_the_reduced_embedding_vector_global)
	print("here19")
	# print('gradients_mem_addr: ', gradients_mem_addr)
	for trace in mem_trace:
		for grad in gradients_mem_addr:
			trace.append((grad, 'W'))
	print("here20")
	# print('mem_trace: ', mem_trace)
	# gradients write back done
	duplicated_grad_addr = [[] for _ in range(batch_num)]
	print("here21")
	start_time = time.time()
	for idx, access in enumerate(gather_op_access):
		for i in range(len(access)):
			duplicated_grad_addr[idx].append(i + max(gradients_mem_addr) + 1)
	end_time = time.time()
	print(f"Time spent: {end_time - start_time:.6f} seconds")
	print("here22")
	# print('duplicated_grad_addr: ', duplicated_grad_addr) 
	# duplicated_grad_addr:  [[20, 21, 22, 23, 24, 25, 26, 27], [20, 21, 22, 23, 24, 25, 26, 27, 28]]

	grad_to_duplicate_access = [[val + min(gradients_mem_addr) for val in sublist] for sublist in entry_to_bag]
	print("here23")
	# print('grad_to_duplicate_access: ', grad_to_duplicate_access)
	# grad_to_duplicate_access:  [[16, 16, 17, 17, 17, 18, 19, 19], [16, 17, 17, 17, 18, 18, 18, 19, 19]]

	# duplicate gradients operation
	for idx, access in enumerate(grad_to_duplicate_access):
		for i in range(len(access)):
			mem_trace[idx].append((grad_to_duplicate_access[idx][i], 'R'))
			mem_trace[idx].append((duplicated_grad_addr[idx][i], 'R'))
			mem_trace[idx].append((duplicated_grad_addr[idx][i], 'W'))
			
		#print(f"Size of mem_trace1: {sys.getsizeof(mem_trace)/(2**20)} MB")
	print("here24")
	# print('mem_trace: ', mem_trace)
	# coalescing gradients
	# gather_op_access == [[2, 4, 0, 2, 4, 7, 9, 10], [2, 3, 4, 5, 7, 8, 10, 8, 11]]

	coalesce_dst = []
	for lst in gather_op_access:
		# Find the unique elements and sort them
		unique_sorted = sorted(set(lst)) # set(lst) removes duplicate values from lst, leaving only unique elements.
		# Create a mapping from element to its rank
		mapping = {value: index for index, value in enumerate(unique_sorted)}
		# Remap the values in the list according to the mapping
		remapped_list = [mapping[value] for value in lst] # remapped_list => indicate how big is the 
		coalesce_dst.append(remapped_list)
	print("here25")
	# print('coalesce_dst: ', coalesce_dst)
	# coalesce_dst:  [[1, 2, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 5, 7]]

	coalesce_dst_addr = []
	for idx, lst in enumerate(coalesce_dst):
		max_addr = max(duplicated_grad_addr[idx])
		coalesce_dst_addr.append([num + max_addr + 1 for num in lst])
	print("here26")
	# print('coalesce_dst_addr:', coalesce_dst_addr)
	# coalesce_dst_addr: [[29, 30, 28, 29, 30, 31, 32, 33], [29, 30, 31, 32, 33, 34, 35, 34, 36]]
	# duplicated_grad_addr:  [[20, 21, 22, 23, 24, 25, 26, 27], [20, 21, 22, 23, 24, 25, 26, 27, 28]]
	
	#coalesce operation
	for idx, access in enumerate(duplicated_grad_addr):
		for i in range(len(access)):
			mem_trace[idx].append((duplicated_grad_addr[idx][i], 'R'))
			mem_trace[idx].append((coalesce_dst_addr[idx][i], 'R'))
			mem_trace[idx].append((coalesce_dst_addr[idx][i], 'W'))
		#print(f"Size of mem_trace2: {sys.getsizeof(mem_trace)/(2**20)} MB")
	print("here27")
	# print('mem_trace: ', mem_trace)

	write_back_to_table = [sorted(set(lst)) for lst in gather_op_access]
	print("here28")
	# print('write_back_to_table: ', write_back_to_table)
	# write_back_to_table:  [[0, 2, 4, 7, 9, 10], [2, 3, 4, 5, 7, 8, 10, 11]]
	coalesce_grad_ready_to_write_back = [sorted(set(lst)) for lst in coalesce_dst_addr]
	print("here29")
	# print('coalesce_grad_ready_to_write_back: ', coalesce_grad_ready_to_write_back)
	# coalesce_grad_ready_to_write_back:  [[28, 29, 30, 31, 32, 33], [29, 30, 31, 32, 33, 34, 35, 36]]
	
	#update emb table with coalesced gradients
	for idx, access in enumerate(write_back_to_table):
		for i in range(len(access)):
			mem_trace[idx].append((coalesce_grad_ready_to_write_back[idx][i], 'R'))
			mem_trace[idx].append((write_back_to_table[idx][i], 'R'))
			mem_trace[idx].append((write_back_to_table[idx][i], 'W'))
		#print(f"Size of mem_trace3: {sys.getsizeof(mem_trace)/(2**20)} MB")
	print("here30")
	# print('standard mem_trace: ', mem_trace)

	memory_needed = [max(lst) for lst in coalesce_dst_addr]
	#print(f"Size of memory_needed: {sys.getsizeof(memory_needed)/(2**20)} MB")
	print("here31")
	# print('memory_needed: ' , memory_needed)

	add_op_count = []
	xxy = [1,2]

	for batch in mem_trace:
		add_count = 0
		for i in range(len(batch) - 1):
			if (batch[i][1] == 'R') and (batch[i+1][1] == 'R'):
				add_count += 1
		add_op_count.append(add_count)
	#print('add_op_count_standard: ' , add_op_count)
	print("here32")

	return mem_trace, memory_needed

def memory_mapping(memory_trace, memory_needed, embedding_table_dimension_global):
	base_address = 0x10000000  # base
	address_shift_per_embedding_vector = 1 * embedding_table_dimension_global # the amount of address shift I need to take next vector

	address_and_action_pair = [[] for _ in range(len(memory_trace))]
	for idx, trace in enumerate(memory_trace):
		all_address = [hex(base_address + address_shift_per_embedding_vector * i) for i in range(memory_needed[idx] + 1)]
		# print("all_address", all_address)
		for item in trace:
			index, action = item
			address = all_address[index]
			address_and_action_pair[idx].append((address, action))
		# address_and_action_pair[idx].append("STOP")
	
	return address_and_action_pair



def write_output_to_txt(address_and_action_pair,finish):
	
	with open(output_file_path, 'a') as file:
		for sublist in address_and_action_pair:
			for item in sublist:
				if item[1] == 'W':  # Check if access type is 'W'
					file.write(f"{item[0]} {item[1]}\n")  # Write with space
				else:
					file.write(f"{item[0]} {item[1]}\n")  # Write with space
		if (finish):
			file.write('STOP\n')  # Write STOP after each sublist


			

def access_count_compare(memtrace0, memtrace1):
	res = [[] for _ in range(2)]
	for i in range(len(memtrace0)):
		res[0].append(len(memtrace0[i]))
		res[1].append(len(memtrace1[i]))
	# print(res)

exc = getattr(builtins, "IOError", "FileNotFoundError")

class LRPolicyScheduler(_LRScheduler):
	def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
		self.num_warmup_steps = num_warmup_steps
		self.decay_start_step = decay_start_step
		self.decay_end_step = decay_start_step + num_decay_steps
		self.num_decay_steps = num_decay_steps

		if self.decay_start_step < self.num_warmup_steps:
			sys.exit("Learning rate warmup must finish before the decay starts")

		super(LRPolicyScheduler, self).__init__(optimizer)

	def get_lr(self):
		step_count = self._step_count
		if step_count < self.num_warmup_steps:
			# warmup
			scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
			lr = [base_lr * scale for base_lr in self.base_lrs]
			self.last_lr = lr
		elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
			# decay
			decayed_steps = step_count - self.decay_start_step
			scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
			min_lr = 0.0000001
			lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
			self.last_lr = lr
		else:
			if self.num_decay_steps > 0:
				# freeze at last, either because we're after decay
				# or because we're between warmup and decay
				lr = self.last_lr
			else:
				# do not adjust
				lr = self.base_lrs
		return lr

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
	def create_mlp(self, ln, sigmoid_layer):
		# build MLP layer by layer
		layers = nn.ModuleList()
		for i in range(0, ln.size - 1):
			n = ln[i]
			m = ln[i + 1]

			# construct fully connected operator
			LL = nn.Linear(int(n), int(m), bias=True)

			# initialize the weights
			# with torch.no_grad():
			# custom Xavier input, output or two-sided fill
			mean = 0.0  # std_dev = np.sqrt(variance)
			std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
			W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
			std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
			bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
			# approach 1
			LL.weight.data = torch.tensor(W, requires_grad=True)
			LL.bias.data = torch.tensor(bt, requires_grad=True)
			# approach 2
			# LL.weight.data.copy_(torch.tensor(W))
			# LL.bias.data.copy_(torch.tensor(bt))
			# approach 3
			# LL.weight = Parameter(torch.tensor(W),requires_grad=True)
			# LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
			layers.append(LL)

			# construct sigmoid or relu operator
			if i == sigmoid_layer:
				layers.append(nn.Sigmoid())
			else:
				layers.append(nn.ReLU())

		# approach 1: use ModuleList
		# return layers
		# approach 2: use Sequential container to wrap all layers
		return torch.nn.Sequential(*layers)

	def create_emb(self, m, ln):
		emb_l = nn.ModuleList()
		for i in range(0, ln.size):
			n = ln[i]
			# construct embedding operator
			if self.qr_flag and n > self.qr_threshold:
				EE = QREmbeddingBag(n, m, self.qr_collisions,
					operation=self.qr_operation, mode="sum", sparse=True)
			elif self.md_flag:
				base = max(m)
				_m = m[i] if n > self.md_threshold else base
				EE = PrEmbeddingBag(n, _m, base)
				# use np initialization as below for consistency...
				W = np.random.uniform(
					low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
				).astype(np.float32)
				EE.embs.weight.data = torch.tensor(W, requires_grad=True)

			else:
				EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)

				# initialize embeddings
				# nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
				W = np.random.uniform(
					low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
				).astype(np.float32)
				# approach 1
				EE.weight.data = torch.tensor(W, requires_grad=True)
				# approach 2
				# EE.weight.data.copy_(torch.tensor(W))
				# approach 3
				# EE.weight = Parameter(torch.tensor(W),requires_grad=True)

			emb_l.append(EE)

		return emb_l

	def __init__(
		self,
		m_spa=None,
		ln_emb=None,
		ln_bot=None,
		ln_top=None,
		arch_interaction_op=None,
		arch_interaction_itself=False,
		sigmoid_bot=-1,
		sigmoid_top=-1,
		sync_dense_params=True,
		loss_threshold=0.0,
		ndevices=-1,
		qr_flag=False,
		qr_operation="mult",
		qr_collisions=0,
		qr_threshold=200,
		md_flag=False,
		md_threshold=200,
	):
		super(DLRM_Net, self).__init__()

		if (
			(m_spa is not None)
			and (ln_emb is not None)
			and (ln_bot is not None)
			and (ln_top is not None)
			and (arch_interaction_op is not None)
		):

			# save arguments
			self.ndevices = ndevices
			self.output_d = 0
			self.parallel_model_batch_size = -1
			self.parallel_model_is_not_prepared = True
			self.arch_interaction_op = arch_interaction_op
			self.arch_interaction_itself = arch_interaction_itself
			self.sync_dense_params = sync_dense_params
			self.loss_threshold = loss_threshold
			# create variables for QR embedding if applicable
			self.qr_flag = qr_flag
			if self.qr_flag:
				self.qr_collisions = qr_collisions
				self.qr_operation = qr_operation
				self.qr_threshold = qr_threshold
			# create variables for MD embedding if applicable
			self.md_flag = md_flag
			if self.md_flag:
				self.md_threshold = md_threshold
			
			self.emb_l = self.create_emb(m_spa, ln_emb)
			# print("EMB : ", ln_emb)
			self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
			self.bot_l = self.bot_l.to("cuda:0")
			self.top_l = self.create_mlp(ln_top, sigmoid_top)
			self.top_l = self.top_l.to("cuda:0")

	def apply_mlp(self, x, layers):
		# approach 1: use ModuleList
		# for layer in layers:
		#     x = layer(x)
		# return x
		# approach 2: use Sequential container to wrap all layers
		return layers(x)
	

	def apply_emb(self, lS_o, lS_i, emb_l):   # lS_i : list of sparse indices (embedding vector indices)  lS_o : list of sparse offset
		#print("apply_embbbbbbbbbbbbbb")
		# global size_of_the_reduced_embedding_vector_global
		# global offset_global
		# WARNING: notice that we are processing the batch at once. We implicitly
		# assume that the data is laid out such that:
		# 1. each embedding is indexed with a group of sparse indices,
		#   corresponding to a single lookup
		# 2. for each embedding the lookups are further organized into a batch
		# 3. for a list of embedding tables there is a list of batched lookups
		# global embedding_table_gather_reduce_access
		# if 'embedding_table_gather_reduce_access' not in globals(): # if not exist, initiate it as a empty list
		# 	embedding_table_gather_reduce_access = []
		#embedding_table_gather_reduce_access = []
		# ly = []
		# if 'offset_global' not in globals():
		# 	offset_global = []


		#offset_global = []        
		#offset_global.append(lS_o)
		
		
		# size = sys.getsizeof( embedding_table_gather_reduce_access)
		# for item in  embedding_table_gather_reduce_access:
		# 	size += sys.getsizeof(item)
		
		# size2 = sys.getsizeof( offset_global)
		# for item2 in  offset_global:
		# 	size2 += sys.getsizeof(item2)

		# print(f"Size of embedding_table_gather_reduce_access: {size/(2**20)} MB")
		# print(f"Size of offset_global: {size2/(2**20)} MB")

		
		ly = []
		
		
		
		#for k, sparse_index_group_batch in enumerate(lS_i):
		for k in range(len(lS_i)):
			
			#print("lS_i = ",lS_i)
			#print("lS_o",lS_o)	
			#finish = (k==(len(lS_i)-1))
			
			# embedding lookup
			# We are using EmbeddingBag, which implicitly uses sum operator.
			# The embeddings are represented as tall matrices, with sum
			# happening vertically across 0 axis, resulting in a row vector
			sparse_index_group_batch = lS_i[k]
			sparse_offset_group_batch = lS_o[k]
			E = emb_l[k]
			
			V = E(sparse_index_group_batch, sparse_offset_group_batch)

			# size_of_the_reduced_embedding_vector_global = V.size(0)
			# embedding_table_gather_reduce_access = [[0, [2, 4, 0, 2, 4]], [1, [1, 3, 4]], [0, [2, 3, 4, 5]], [1, [1, 2, 4, 2, 5]]]
			# offset_global  = [[[0, 2], [0, 1]], [[0, 1], [0, 3]]]
			# print("embedding_ga_re: ",embedding_table_gather_reduce_access)
			# offset_global.append(lS_o[k])
			# embedding_table_gather_reduce_access = [[0, [2, 4, 0, 2, 4]]]
			# offset_global  = [[[0, 2]]]
			# size_of_the_reduced_embedding_vector_global = 4
			#size_of_the_reduced_embedding_vector_global = V.size(0)

			# print("embedding_ga_re: ",embedding_table_gather_reduce_access)
			# print("embedding_table_gather_reduce_access", embedding_table_gather_reduce_access)
			# print("embedding_table_len_global",embedding_table_len_global)
			# print("size_of_the_reduced_embedding_vector_global",size_of_the_reduced_embedding_vector_global)
			# print("offset_global",offset_global)
			
			# with open(output_file_path, 'a') as file:
			# 		# You can do an initial write here if needed
			# 		file.write("End of a lookup.\n")
			ly.append(V)

		embedding_table_gather_reduce_access = []
		offset_global = lS_o
		size_of_the_reduced_embedding_vector_global = 0    
		for i in range(len(lS_i)):
			sparse_index_group_batch = lS_i[i]
			sparse_offset_group_batch = lS_o[i]
			# offset_global.append(lS_o[i].unsqueeze(0))
			embedding_table_gather_reduce_access.append([i, sparse_index_group_batch])
		print("embedding_table_gather_reduce_access length",len(embedding_table_gather_reduce_access[0][1]))
		print("embedding_table_gather_reduce_access",embedding_table_gather_reduce_access)
		#print("embedding_table_gather_reduce_access",embedding_table_gather_reduce_access)
		#print("offset_global",offset_global)
		size_of_the_reduced_embedding_vector_global = sum([t.size(0) for t in ly])
		#print("size_of_the_reduced_embedding_vector_global: ",size_of_the_reduced_embedding_vector_global)
		memory_trace, memory_needed = training_trace_standard(embedding_table_gather_reduce_access, embedding_table_len_global, size_of_the_reduced_embedding_vector_global, offset_global)
		
		address_and_action_pair = memory_mapping(memory_trace, memory_needed, embedding_table_dimension_global)
		print("Writing Trace File...")
		write_output_to_txt(address_and_action_pair,True)
		#print("Finish Writing one batch of trace !!! ")
		#print("ly",ly)
		
		return ly

	def interact_features(self, x, ly):
		if self.arch_interaction_op == "dot":
			# concatenate dense and sparse features
			(batch_size, d) = x.shape
			T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
			# perform a dot product
			Z = torch.bmm(T, torch.transpose(T, 1, 2))
			# append dense feature with the interactions (into a row vector)
			# approach 1: all
			# Zflat = Z.view((batch_size, -1))
			# approach 2: unique
			_, ni, nj = Z.shape
			# approach 1: tril_indices
			# offset = 0 if self.arch_interaction_itself else -1
			# li, lj = torch.tril_indices(ni, nj, offset=offset)
			# approach 2: custom
			offset = 1 if self.arch_interaction_itself else 0
			li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
			lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
			Zflat = Z[:, li, lj]
			# concatenate dense features and interactions
			R = torch.cat([x] + [Zflat], dim=1)
		elif self.arch_interaction_op == "cat":
			# concatenation features (into a row vector)
			R = torch.cat([x] + ly, dim=1)
		else:
			sys.exit(
				"ERROR: --arch-interaction-op="
				+ self.arch_interaction_op
				+ " is not supported"
			)

		return R

	def forward(self, dense_x, lS_o, lS_i):
		return self.mixed_forward(dense_x, lS_o, lS_i)
		

	def mixed_forward(self, dense_x, lS_o, lS_i):	
		# Process dense features on GPU in a data parallel fashion
		### prepare model (overwrite) ###
		# WARNING: # of devices must be >= batch size in parallel_forward call
		batch_size = dense_x.size()[0]
		ndevices = min(self.ndevices, batch_size, len(self.emb_l))
		device_ids = range(ndevices)
		# WARNING: must redistribute the model if mini-batch size changes(this is common
		# for last mini-batch, when # of elements in the dataset/batch size is not even
		if self.parallel_model_batch_size != batch_size:
			self.parallel_model_is_not_prepared = True

		if self.parallel_model_is_not_prepared or self.sync_dense_params:
			# replicate mlp (data parallelism)
			self.bot_l_replicas = replicate(self.bot_l, device_ids)
			self.top_l_replicas = replicate(self.top_l, device_ids)
			self.parallel_model_batch_size = batch_size

		### prepare input (overwrite) ###
		# scatter dense features (data parallelism)
		# print(dense_x.device)
		dense_x = scatter(dense_x, device_ids, dim=0)

		x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)

		# process sparse features(using embeddings) on CPU, resulting in a list of row vectors
		ly = self.apply_emb(lS_o, lS_i, self.emb_l)
		# for y in ly:
		#     print(y.detach().cpu().numpy())

		ly = torch.stack(ly)

		# scattering ly across GPU's
		ly = ly.to("cuda:0")
		t_list = []
		for k, _ in enumerate(self.emb_l):
			y = scatter(ly[k], device_ids, dim=0)
			t_list.append(y)
		# adjust the list to be ordered per device
		ly = list(map(lambda y: list(y), zip(*t_list)))
		
		# interactions
		z = []
		for k in range(ndevices):
			zk = self.interact_features(x[k], ly[k])
			z.append(zk)
		# debug prints
		# print(z)

		# top mlp
		# WARNING: Note that the self.top_l is a list of top mlp modules that
		# have been replicated across devices, while z is a list of interaction results
		# that by construction are scattered across devices on the first (batch) dim.
		# The output is a list of tensors scattered across devices according to the
		# distribution of z.
		p = parallel_apply(self.top_l_replicas, z, None, device_ids)

		begin_gather = time_wrap(use_gpu)
		### gather the distributed results ###
		p0 = gather(p, self.output_d, dim=0)
			
		# clamp output if needed
		if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
			z0 = torch.clamp(
				p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
			)
		else:
			z0 = p0

		return z0

def dash_separated_ints(value):
	vals = value.split('-')
	for val in vals:
		try:
			int(val)
		except ValueError:
			raise argparse.ArgumentTypeError(
				"%s is not a valid dash separated list of ints" % value)

	return value


def dash_separated_floats(value):
	vals = value.split('-')
	for val in vals:
		try:
			float(val)
		except ValueError:
			raise argparse.ArgumentTypeError(
				"%s is not a valid dash separated list of floats" % value)

	return value


if __name__ == "__main__":
	### import packages ###
	import sys
	import argparse

	### parse arguments ###
	parser = argparse.ArgumentParser(
		description="Train Deep Learning Recommendation Model (DLRM)"
	)
	# model related parameters
	parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
	parser.add_argument(
		"--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
	# j will be replaced with the table number
	parser.add_argument(
		"--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
	parser.add_argument(
		"--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
	parser.add_argument(
		"--arch-interaction-op", type=str, choices=['dot', 'cat'], default="dot")
	parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
	# embedding table options
	parser.add_argument("--md-flag", action="store_true", default=False)
	parser.add_argument("--md-threshold", type=int, default=200)
	parser.add_argument("--md-temperature", type=float, default=0.3)
	parser.add_argument("--md-round-dims", action="store_true", default=False)
	parser.add_argument("--qr-flag", action="store_true", default=False)
	parser.add_argument("--qr-threshold", type=int, default=200)
	parser.add_argument("--qr-operation", type=str, default="mult")
	parser.add_argument("--qr-collisions", type=int, default=4)
	# activations and loss
	parser.add_argument("--activation-function", type=str, default="relu")
	parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
	parser.add_argument(
		"--loss-weights", type=dash_separated_floats, default="1.0-1.0")  # for wbce
	parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
	parser.add_argument("--round-targets", type=bool, default=False)
	# data
	parser.add_argument("--data-size", type=int, default=1)
	parser.add_argument("--num-batches", type=int, default=0)
	parser.add_argument(
		"--data-generation", type=str, default="random"
	)  # synthetic or dataset
	parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
	parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
	parser.add_argument("--raw-data-file", type=str, default="")
	parser.add_argument("--processed-data-file", type=str, default="")
	parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
	parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
	parser.add_argument("--max-ind-range", type=int, default=-1)
	parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
	parser.add_argument("--num-indices-per-lookup", type=int, default=10)
	parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--memory-map", action="store_true", default=False)
	parser.add_argument("--dataset-multiprocessing", action="store_true", default=False,
						help="The Kaggle dataset can be multiprocessed in an environment \
						with more than 7 CPU cores and more than 20 GB of memory. \n \
						The Terabyte dataset can be multiprocessed in an environment \
						with more than 24 CPU cores and at least 1 TB of memory.")
	# training
	parser.add_argument("--mini-batch-size", type=int, default=1)
	parser.add_argument("--nepochs", type=int, default=1)
	parser.add_argument("--learning-rate", type=float, default=0.01)
	parser.add_argument("--print-precision", type=int, default=5)
	parser.add_argument("--numpy-rand-seed", type=int, default=123)
	parser.add_argument("--sync-dense-params", type=bool, default=True)
	# inference
	parser.add_argument("--inference-only", action="store_true", default=False)
	# onnx
	parser.add_argument("--save-onnx", action="store_true", default=False)
	# gpu
	parser.add_argument("--use-gpu", action="store_true", default=True)
	# debugging and profiling
	parser.add_argument("--print-freq", type=int, default=1)
	parser.add_argument("--test-freq", type=int, default=-1)
	parser.add_argument("--test-mini-batch-size", type=int, default=-1)
	parser.add_argument("--test-num-workers", type=int, default=-1)
	parser.add_argument("--print-time", action="store_true", default=False)
	parser.add_argument("--debug-mode", action="store_true", default=False)
	parser.add_argument("--enable-profiling", action="store_true", default=False)
	parser.add_argument("--plot-compute-graph", action="store_true", default=False)
	# store/load model
	parser.add_argument("--save-model", type=str, default="")
	parser.add_argument("--load-model", type=str, default="")
	# mlperf logging (disables other output and stops early)
	parser.add_argument("--mlperf-logging", action="store_true", default=False)
	# stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
	parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
	# stop at target AUC Terabyte (no subsampling) 0.8025
	parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
	parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
	parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
	# LR policy
	parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
	parser.add_argument("--lr-decay-start-step", type=int, default=0)
	parser.add_argument("--lr-num-decay-steps", type=int, default=0)
	# output file
	parser.add_argument("--output-trace-file", type=str, default="random_whole_store")
	
	args = parser.parse_args()

	if args.mlperf_logging:
		print('command line args: ', json.dumps(vars(args)))

	### some basic setup ###
	np.random.seed(args.numpy_rand_seed)
	np.set_printoptions(precision=args.print_precision)
	torch.set_printoptions(precision=args.print_precision)
	torch.manual_seed(args.numpy_rand_seed)

	if (args.test_mini_batch_size < 0):
		# if the parameter is not set, use the training batch size
		args.test_mini_batch_size = args.mini_batch_size
	if (args.test_num_workers < 0):
		# if the parameter is not set, use the same parameter for training
		args.test_num_workers = args.num_workers

	use_gpu = args.use_gpu and torch.cuda.is_available()
	if use_gpu:
		torch.cuda.manual_seed_all(args.numpy_rand_seed)
		torch.backends.cudnn.deterministic = True
		device = torch.device("cuda", 0)
		ngpus = torch.cuda.device_count()  # 1
		print("Running DLRM Baseline")
		print("Using CPU and {} GPU(s)...".format(ngpus))
	else:
		device = torch.device("cpu")
		print("Using CPU...")

	### prepare training data ###
	ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
	# input data
	if (args.data_generation == "dataset"):

		train_data, train_ld, test_data, test_ld = \
			dp.make_criteo_data_and_loaders(args)
		nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
		nbatches_test = len(test_ld)

		ln_emb = train_data.counts
		# enforce maximum limit on number of vectors per embedding
		if args.max_ind_range > 0:
			ln_emb = np.array(list(map(
				lambda x: x if x < args.max_ind_range else args.max_ind_range,
				ln_emb
			)))
		m_den = train_data.m_den
		ln_bot[0] = m_den
	else:
		# input and target at random
		ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
		m_den = ln_bot[0]
		train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
		nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

	### parse command line arguments ###
	m_spa = args.arch_sparse_feature_size
	global embedding_table_dimension_global
	embedding_table_dimension_global = m_spa
	ln_emb = np.asarray(ln_emb)
	global embedding_table_len_global
	embedding_table_len_global = ln_emb
	#embedding_table_len_global = [6,6]
	
	num_fea = ln_emb.size + 1  # num sparse + num dense features
    #     embedding table dimension
    #   _ |-------------------------|
    # l | **************************
    # e | **************************
    # n | **************************
    # g | **************************
    # t | **************************
    # h | **************************
    #   —

	m_den_out = ln_bot[ln_bot.size - 1]
	if args.arch_interaction_op == "dot":
		# approach 1: all
		# num_int = num_fea * num_fea + m_den_out
		# approach 2: unique
		if args.arch_interaction_itself:
			num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
		else:
			num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
	elif args.arch_interaction_op == "cat":
		num_int = num_fea * m_den_out
	else:
		sys.exit(
			"ERROR: --arch-interaction-op="
			+ args.arch_interaction_op
			+ " is not supported"
		)
	arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
	ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

	# sanity check: feature sizes and mlp dimensions must match
	if m_den != ln_bot[0]:
		sys.exit(
			"ERROR: arch-dense-feature-size "
			+ str(m_den)
			+ " does not match first dim of bottom mlp "
			+ str(ln_bot[0])
		)
	if args.qr_flag:
		if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
			sys.exit(
				"ERROR: 2 arch-sparse-feature-size "
				+ str(2 * m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
				+ " (note that the last dim of bottom mlp must be 2x the embedding dim)"
			)
		if args.qr_operation != "concat" and m_spa != m_den_out:
			sys.exit(
				"ERROR: arch-sparse-feature-size "
				+ str(m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
			)
	else:
		if m_spa != m_den_out:
			sys.exit(
				"ERROR: arch-sparse-feature-size "
				+ str(m_spa)
				+ " does not match last dim of bottom mlp "
				+ str(m_den_out)
			)
	if num_int != ln_top[0]:
		sys.exit(
			"ERROR: # of feature interactions "
			+ str(num_int)
			+ " does not match first dimension of top mlp "
			+ str(ln_top[0])
		)

	# assign mixed dimensions if applicable
	if args.md_flag:
		m_spa = md_solver(
			torch.tensor(ln_emb),
			args.md_temperature,  # alpha
			d0=m_spa,
			round_dim=args.md_round_dims
		).tolist()

	# test prints (model arch)
	if args.debug_mode:
		print("model arch:")
		print(
			"mlp top arch "
			+ str(ln_top.size - 1)
			+ " layers, with input to output dimensions:"
		)
		print(ln_top)
		print("# of interactions")
		print(num_int)
		print(
			"mlp bot arch "
			+ str(ln_bot.size - 1)
			+ " layers, with input to output dimensions:"
		)
		print(ln_bot)
		print("# of features (sparse and dense)")
		print(num_fea)
		print("dense feature size")
		print(m_den)
		print("sparse feature size")
		print(m_spa)
		print(
			"# of embeddings (= # of sparse features) "
			+ str(ln_emb.size)
			+ ", with dimensions "
			+ str(m_spa)
			+ "x:"
		)
		print(ln_emb)

		print("data (inputs and targets):")
		numer_of_batches = 0

		def modify_batches():
			global numer_of_batches  

		modify_batches()
		#for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
		for j, (X, lS_o, lS_i, T) in enumerate(tqdm(train_ld, desc="Training Progress", total=len(train_ld))):
			
			# early exit if nbatches was set by the user and has been exceeded
			if nbatches > 0 and j >= nbatches:
				break
			numer_of_batches += 1
			print("numer_of_batches: ",numer_of_batches)
			print("mini-batch: %d" % j)
			print(X.detach().cpu().numpy())
			# transform offsets to lengths when printing
			print(
				[
					np.diff(
						S_o.detach().cpu().tolist() + list(lS_i[i].shape)
					).tolist()
					for i, S_o in enumerate(lS_o)
				]
			)
			print([S_i.detach().cpu().tolist() for S_i in lS_i])
			print(T.detach().cpu().numpy())

	ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

	### construct the neural network specified above ###
	# WARNING: to obtain exactly the same initialization for
	# the weights we need to start from the same random seed.
	# np.random.seed(args.numpy_rand_seed)
	dlrm = DLRM_Net(
		m_spa,
		ln_emb,
		ln_bot,
		ln_top,
		arch_interaction_op=args.arch_interaction_op,
		arch_interaction_itself=args.arch_interaction_itself,
		sigmoid_bot=-1,
		sigmoid_top=ln_top.size - 2,
		sync_dense_params=args.sync_dense_params,
		loss_threshold=args.loss_threshold,
		ndevices=ndevices,
		qr_flag=args.qr_flag,
		qr_operation=args.qr_operation,
		qr_collisions=args.qr_collisions,
		qr_threshold=args.qr_threshold,
		md_flag=args.md_flag,
		md_threshold=args.md_threshold,
	)
	# test prints
	# if args.debug_mode:
	# 	print("initial parameters (weights and bias):")
	# 	for param in dlrm.parameters():
	# 		print(param.detach().cpu().numpy())
		# print(dlrm)

	# specify the loss function
	if args.loss_function == "mse":
		loss_fn = torch.nn.MSELoss(reduction="mean")
	elif args.loss_function == "bce":
		loss_fn = torch.nn.BCELoss(reduction="mean")
	elif args.loss_function == "wbce":
		loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
		loss_fn = torch.nn.BCELoss(reduction="none")
	else:
		sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

	if not args.inference_only:
		# specify the optimizer algorithm
		optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
		lr_scheduler = LRPolicyScheduler(optimizer, args.lr_num_warmup_steps, args.lr_decay_start_step,
										 args.lr_num_decay_steps)

	### main loop ###
	def time_wrap(use_gpu):
		if use_gpu:
			torch.cuda.synchronize()
		return time.time()

	def dlrm_wrap(X, lS_o, lS_i, use_gpu, device):
		if use_gpu:  # .cuda()
			return dlrm(
				X.to(device),
				lS_o,
				lS_i
			)
		else:
			return dlrm(X, lS_o, lS_i)

	def loss_fn_wrap(Z, T, use_gpu, device):
		if args.loss_function == "mse" or args.loss_function == "bce":
			if use_gpu:
				return loss_fn(Z, T.to(device))
			else:
				return loss_fn(Z, T)
		elif args.loss_function == "wbce":
			if use_gpu:
				loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
				loss_fn_ = loss_fn(Z, T.to(device))
			else:
				loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
				loss_fn_ = loss_fn(Z, T.to(device))
			loss_sc_ = loss_ws_ * loss_fn_
			# debug prints
			# print(loss_ws_)
			# print(loss_fn_)
			return loss_sc_.mean()

	# training or inference
	best_gA_test = 0
	best_auc_test = 0
	skip_upto_epoch = 0
	skip_upto_batch = 0
	total_time = 0
	total_loss = 0
	total_accu = 0
	total_iter = 0
	total_samp = 0
	forward_time = 0
	backward_time = 0
	optimizer_time = 0
	scheduler_time = 0
	k = 0
	stop = 0

	# Load model is specified
	if not (args.load_model == ""):
		print("Loading saved model {}".format(args.load_model))
		if use_gpu:
			if dlrm.ndevices > 1:
				# NOTE: when targeting inference on multiple GPUs,
				# load the model as is on CPU or GPU, with the move
				# to multiple GPUs to be done in parallel_forward
				ld_model = torch.load(args.load_model)
			else:
				# NOTE: when targeting inference on single GPU,
				# note that the call to .to(device) has already happened
				ld_model = torch.load(
					args.load_model,
					map_location=torch.device('cuda')
					# map_location=lambda storage, loc: storage.cuda(0)
				)
		else:
			# when targeting inference on CPU
			ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
		dlrm.load_state_dict(ld_model["state_dict"])
		ld_j = ld_model["iter"]
		ld_k = ld_model["epoch"]
		ld_nepochs = ld_model["nepochs"]
		ld_nbatches = ld_model["nbatches"]
		ld_nbatches_test = ld_model["nbatches_test"]
		ld_gA = ld_model["train_acc"]
		ld_gL = ld_model["train_loss"]
		ld_total_loss = ld_model["total_loss"]
		ld_total_accu = ld_model["total_accu"]
		ld_gA_test = ld_model["test_acc"]
		ld_gL_test = ld_model["test_loss"]
		if not args.inference_only:
			optimizer.load_state_dict(ld_model["opt_state_dict"])
			best_gA_test = ld_gA_test
			total_loss = ld_total_loss
			total_accu = ld_total_accu
			skip_upto_epoch = ld_k  # epochs
			skip_upto_batch = ld_j  # batches
		else:
			args.print_freq = ld_nbatches
			args.test_freq = 0

		print(
			"Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
				ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
			)
		)
		print(
			"Training state: loss = {:.6f}, accuracy = {:3.3f} %".format(
				ld_gL, ld_gA * 100
			)
		)
		print(
			"Testing state: loss = {:.6f}, accuracy = {:3.3f} %".format(
				ld_gL_test, ld_gA_test * 100
			)
		)

	print("time/loss/accuracy (if enabled):")
	with torch.autograd.profiler.profile(enabled=args.enable_profiling, use_cuda=use_gpu) as prof:
		while k < args.nepochs:
			if k==0:
				with open(output_file_path, 'w') as file:
					# You can do an initial write here if needed
					file.write("Starting new file.\n")
			if k < skip_upto_epoch:
				continue

			if stop == 1:
				break

			accum_time_begin = time_wrap(use_gpu)

			if args.mlperf_logging:
				previous_iteration_time = None

			# for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
			for j, (X, lS_o, lS_i, T) in enumerate(tqdm(train_ld, desc="Training Progress", total=len(train_ld))):
				if j == 0 and args.save_onnx:
					(X_onnx, lS_o_onnx, lS_i_onnx) = (X, lS_o, lS_i)

				if j < skip_upto_batch:
					continue

				if args.mlperf_logging:
					current_time = time_wrap(use_gpu)
					if previous_iteration_time:
						iteration_time = current_time - previous_iteration_time
					else:
						iteration_time = 0
					previous_iteration_time = current_time
				else:
					t1 = time_wrap(use_gpu)

				# early exit if nbatches was set by the user and has been exceeded
				if nbatches > 0 and j >= nbatches:
					break
				'''
				# debug prints
				print("input and targets")
				print(X.detach().cpu().numpy())
				print([np.diff(S_o.detach().cpu().tolist()
					   + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
				print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
				print(T.detach().cpu().numpy())
				'''
				should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)

				begin_forward = time_wrap(use_gpu)
				# forward pass
				Z = dlrm_wrap(X, lS_o, lS_i, use_gpu, device)

				end_forward = time_wrap(use_gpu)

				# loss
				E = loss_fn_wrap(Z, T, use_gpu, device)
				'''
				# debug prints
				print("output and loss")
				print(Z.detach().cpu().numpy())
				print(E.detach().cpu().numpy())
				'''
				# compute loss and accuracy
				L = E.detach().cpu().numpy()  # numpy array
				S = Z.detach().cpu().numpy()  # numpy array
				T = T.detach().cpu().numpy()  # numpy array
				mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
				A = np.sum((np.round(S, 0) == T).astype(np.uint8))

				if not args.inference_only:
					# scaled error gradient propagation
					# (where we do not accumulate gradients across mini-batches)
					optimizer.zero_grad()
					# backward pass
					E.backward()
					# for name, param in dlrm.named_parameters():
					# 	print(f"Parameter: {name}")
					# 	print(f"  Shape: {param.shape}")
					# 	print(f"  Gradient:\n{param.grad}")
					# 	print()
					# debug prints (check gradient norm)
					# for l in mlp.layers:
					#     if hasattr(l, 'weight'):
					#          print(l.weight.grad.norm().item())
					end_backward = time_wrap(use_gpu)

					# optimizer
					optimizer.step()

					end_optimizing = time_wrap(use_gpu)

					lr_scheduler.step()

					end_scheduling = time_wrap(use_gpu)

				if args.mlperf_logging:
					total_time += iteration_time
				else:
					t2 = time_wrap(use_gpu)
					total_time += t2 - t1
				total_accu += A
				total_loss += L * mbs
				total_iter += 1
				total_samp += mbs
				forward_time += end_forward - begin_forward
				backward_time += end_backward - end_forward
				optimizer_time += end_optimizing - end_backward
				scheduler_time += end_scheduling - end_optimizing

				#should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
				should_test = (
					(args.test_freq > 0)
					and (args.data_generation == "dataset")
					and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
				)

				# print time, loss and accuracy
				if should_print or should_test:
					gT = 1000.0 * total_time / total_iter if args.print_time else -1
					total_time = 0

					gA = total_accu / total_samp
					total_accu = 0

					gL = total_loss / total_samp
					total_loss = 0

					gForward = 1000 * forward_time / total_iter

					gBackward = 1000 * backward_time / total_iter

					gOptimizer = 1000 * optimizer_time / total_iter

					gScheduler = 1000 * scheduler_time /total_iter

					str_run_type = "inference" if args.inference_only else "training"

					print("Forward ", gForward)
					print("Backward ", gBackward)
					print("Optimizer ", gOptimizer)
					print("LR_scheduler ", gScheduler)
					print("Epoch ", k)
					print("Iteration ", j+1)
					print("Total_Iterations ", nbatches)
					print("Iteration_time ", gT)
					print("Loss ", gL)
					print("Accuracy ", gA*100)
					print("\n")

					# Uncomment the line below to print out the total time with overhead
					# print("Accumulated time so far: {}" \
					# .format(time_wrap(use_gpu) - accum_time_begin))
					total_iter = 0
					total_samp = 0
					forward_time = 0
					backward_time = 0
					optimizer_time = 0
					scheduler_time = 0

				# testing
				if should_test and not args.inference_only:
					# don't measure training iter time in a test iteration
					if args.mlperf_logging:
						previous_iteration_time = None

					test_accu = 0
					test_loss = 0
					test_samp = 0

					accum_test_time_begin = time_wrap(use_gpu)
					if args.mlperf_logging:
						scores = []
						targets = []

					for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
						# early exit if nbatches was set by the user and was exceeded
						if nbatches > 0 and i >= nbatches:
							break

						t1_test = time_wrap(use_gpu)
						should_print = 0
						# forward pass
						Z_test = dlrm_wrap(
							X_test, lS_o_test, lS_i_test, use_gpu, device
						)
						if args.mlperf_logging:
							S_test = Z_test.detach().cpu().numpy()  # numpy array
							T_test = T_test.detach().cpu().numpy()  # numpy array
							scores.append(S_test)
							targets.append(T_test)
						else:
							# loss
							E_test = loss_fn_wrap(Z_test, T_test, use_gpu, device)

							# compute loss and accuracy
							L_test = E_test.detach().cpu().numpy()  # numpy array
							S_test = Z_test.detach().cpu().numpy()  # numpy array
							T_test = T_test.detach().cpu().numpy()  # numpy array
							mbs_test = T_test.shape[0]  # = mini_batch_size except last
							A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
							test_accu += A_test
							test_loss += L_test * mbs_test
							test_samp += mbs_test

						t2_test = time_wrap(use_gpu)

					if args.mlperf_logging:
						scores = np.concatenate(scores, axis=0)
						targets = np.concatenate(targets, axis=0)

						metrics = {
							'loss' : sklearn.metrics.log_loss,
							'recall' : lambda y_true, y_score:
							sklearn.metrics.recall_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'precision' : lambda y_true, y_score:
							sklearn.metrics.precision_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'f1' : lambda y_true, y_score:
							sklearn.metrics.f1_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							'ap' : sklearn.metrics.average_precision_score,
							'roc_auc' : sklearn.metrics.roc_auc_score,
							'accuracy' : lambda y_true, y_score:
							sklearn.metrics.accuracy_score(
								y_true=y_true,
								y_pred=np.round(y_score)
							),
							# 'pre_curve' : sklearn.metrics.precision_recall_curve,
							# 'roc_curve' :  sklearn.metrics.roc_curve,
						}

						# print("Compute time for validation metric : ", end="")
						# first_it = True
						validation_results = {}
						for metric_name, metric_function in metrics.items():
							# if first_it:
							#     first_it = False
							# else:
							#     print(", ", end="")
							# metric_compute_start = time_wrap(False)
							validation_results[metric_name] = metric_function(
								targets,
								scores
							)
							# metric_compute_end = time_wrap(False)
							# met_time = metric_compute_end - metric_compute_start
							# print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
							#      end="")
						# print(" ms")
						gA_test = validation_results['accuracy']
						gL_test = validation_results['loss']
					else:
						gA_test = test_accu / test_samp
						gL_test = test_loss / test_samp

					is_best = gA_test > best_gA_test
					if is_best:
						best_gA_test = gA_test
						if not (args.save_model == ""):
							print("Saving model to {}".format(args.save_model))
							torch.save(
								{
									"epoch": k,
									"nepochs": args.nepochs,
									"nbatches": nbatches,
									"nbatches_test": nbatches_test,
									"iter": j + 1,
									"state_dict": dlrm.state_dict(),
									"train_acc": gA,
									"train_loss": gL,
									"test_acc": gA_test,
									"test_loss": gL_test,
									"total_loss": total_loss,
									"total_accu": total_accu,
									"opt_state_dict": optimizer.state_dict(),
								},
								args.save_model,
							)

					if args.mlperf_logging:
						is_best = validation_results['roc_auc'] > best_auc_test
						if is_best:
							best_auc_test = validation_results['roc_auc']

						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Epoch ", k)
						print("Test_Loss ", validation_results['loss'])
						print("Test_recall ", validation_results['recall'])
						print("Test_precision ", validation_results['precision'])
						print("Test_f1 ", validation_results['f1'])
						print("Test_ap ", validation_results['ap'])
						print("Test_auc ", validation_results['roc_auc'])
						print("Best_auc ", best_auc_test)
						print("Test_Accuracy ", validation_results['accuracy'] * 100)
						print("Best_Accuracy ", best_gA_test * 100)
						print("\n")

						
					else:
						print("Test_Iteration ", j + 1)
						print("Total_Iterations ", nbatches)
						print("Test_Loss ", gL_test)
						print("Test_Accuracy ", gA_test * 100)
						print("Best_test_Accuracy ", best_gA_test * 100)
						print("\n")
						
					# Uncomment the line below to print out the total time with overhead
					# print("Total test time for this group: {}" \
					# .format(time_wrap(use_gpu) - accum_test_time_begin))

					if (args.mlperf_logging
						and (args.mlperf_acc_threshold > 0)
						and (best_gA_test > args.mlperf_acc_threshold)):
						print("MLPerf testing accuracy threshold "
							  + str(args.mlperf_acc_threshold)
							  + " reached, stop training")
						stop = 1
						break

					if (args.mlperf_logging
						and (args.mlperf_auc_threshold > 0)
						and (best_auc_test > args.mlperf_auc_threshold)):
						print("MLPerf testing auc threshold "
							  + str(args.mlperf_auc_threshold)
							  + " reached, stop training")
						break

			k += 1  # nepochs
		accum_time_end = time_wrap(use_gpu)
		#print("Total_Epoch_Time ", 1000*(accum_time_end - accum_time_begin))

	# profiling
	if args.enable_profiling:
		with open("dlrm_s_pytorch.prof", "w") as prof_f:
			prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
			prof.export_chrome_trace("./dlrm_s_pytorch.json")
		# print(prof.key_averages().table(sort_by="cpu_time_total"))

	# plot compute graph
	if args.plot_compute_graph:
		sys.exit(
			"ERROR: Please install pytorchviz package in order to use the"
			+ " visualization. Then, uncomment its import above as well as"
			+ " three lines below and run the code again."
		)
		# V = Z.mean() if args.inference_only else E
		# dot = make_dot(V, params=dict(dlrm.named_parameters()))
		# dot.render('dlrm_s_pytorch_graph') # write .pdf file

	# test prints
	if not args.inference_only and args.debug_mode:
		print("updated parameters (weights and bias):")
		# for param in dlrm.parameters():
		# 	print(param.detach().cpu().numpy())

	# export the model in onnx
	if args.save_onnx:
		dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
		batch_size = X_onnx.shape[0]
		# debug prints
		# print("batch_size", batch_size)
		# print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
		# print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))

		# force list conversion
		# if torch.is_tensor(lS_o_onnx):
		#    lS_o_onnx = [lS_o_onnx[j] for j in range(len(lS_o_onnx))]
		# if torch.is_tensor(lS_i_onnx):
		#    lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
		# force tensor conversion
		# if isinstance(lS_o_onnx, list):
		#     lS_o_onnx = torch.stack(lS_o_onnx)
		# if isinstance(lS_i_onnx, list):
		#     lS_i_onnx = torch.stack(lS_i_onnx)
		# debug prints
		print("X_onnx.shape", X_onnx.shape)
		if torch.is_tensor(lS_o_onnx):
			print("lS_o_onnx.shape", lS_o_onnx.shape)
		else:
			for oo in lS_o_onnx:
				print("oo.shape", oo.shape)
		if torch.is_tensor(lS_i_onnx):
			print("lS_i_onnx.shape", lS_i_onnx.shape)
		else:
			for ii in lS_i_onnx:
				print("ii.shape", ii.shape)

		# name inputs and outputs
		o_inputs = ["offsets"] if torch.is_tensor(lS_o_onnx) else ["offsets_"+str(i) for i in range(len(lS_o_onnx))]
		i_inputs = ["indices"] if torch.is_tensor(lS_i_onnx) else ["indices_"+str(i) for i in range(len(lS_i_onnx))]
		all_inputs = ["dense_x"] + o_inputs + i_inputs
		#debug prints
		print("inputs", all_inputs)

		# create dynamic_axis dictionaries
		do_inputs = [{'offsets': {1 : 'batch_size' }}] if torch.is_tensor(lS_o_onnx) else [{"offsets_"+str(i) :{0 : 'batch_size'}} for i in range(len(lS_o_onnx))]
		di_inputs = [{'indices': {1 : 'batch_size' }}] if torch.is_tensor(lS_i_onnx) else [{"indices_"+str(i) :{0 : 'batch_size'}} for i in range(len(lS_i_onnx))]
		dynamic_axes = {'dense_x' : {0 : 'batch_size'}, 'pred' : {0 : 'batch_size'}}
		for do in do_inputs:
			dynamic_axes.update(do)
		for di in di_inputs:
			dynamic_axes.update(di)
		# debug prints
		print(dynamic_axes)

		# export model
		torch.onnx.export(
			dlrm, (X_onnx, lS_o_onnx, lS_i_onnx), dlrm_pytorch_onnx_file, verbose=True, use_external_data_format=True, opset_version=11, input_names=all_inputs, output_names=["pred"], dynamic_axes=dynamic_axes
		)
		# recover the model back
		dlrm_pytorch_onnx = onnx.load(dlrm_pytorch_onnx_file)
		# check the onnx model
		onnx.checker.check_model(dlrm_pytorch_onnx)
		'''
		# run model using onnxruntime
		import onnxruntime as rt

		dict_inputs = {}
		dict_inputs["dense_x"] = X_onnx.numpy().astype(np.float32)
		if torch.is_tensor(lS_o_onnx):
			dict_inputs["offsets"] = lS_o_onnx.numpy().astype(np.int64)
		else:
			for i in range(len(lS_o_onnx)):
				dict_inputs["offsets_"+str(i)] = lS_o_onnx[i].numpy().astype(np.int64)
		if torch.is_tensor(lS_i_onnx):
			dict_inputs["indices"] = lS_i_onnx.numpy().astype(np.int64)
		else:
			for i in range(len(lS_i_onnx)):
				dict_inputs["indices_"+str(i)] = lS_i_onnx[i].numpy().astype(np.int64)
		print("dict_inputs", dict_inputs)

		sess = rt.InferenceSession(dlrm_pytorch_onnx_file, rt.SessionOptions())
		prediction = sess.run(output_names=["pred"], input_feed=dict_inputs)
		print("prediction", prediction)
		'''
	# example_mode = False
	# if example_mode == True:
	# 	embedding_table_gather_reduce_access = [[0, [2, 4, 0, 2, 4]], [1, [1, 3, 4]], [0, [2, 3, 4, 5]], [1, [1, 2, 4, 2, 5]]]
	# 	offset_global  = [[[0, 2], [0, 1]], [[0, 1], [0, 3]]]
	
	# def training_trace_standard(embedding_table_gather_reduce_access, embedding_table_len_global, size_of_the_reduced_embedding_vector_global, offset_global):
	# 	print("here")
	# 	total_length = sum(embedding_table_len_global)
	# 	print("here1")
	# 	# memory_index = list(range(total_length)) # 0 ~ total_length-1
	# 	table_size_list = [size for size in embedding_table_len_global]
	# 	print("here2")

	# 	embedding_table_gather_reduce_access = [[elem[0], elem[1].tolist()] for elem in embedding_table_gather_reduce_access] # to list
	# 	print("here3")
	# 	# print("***embedding_table_gather_reduce_access", embedding_table_gather_reduce_access)
	# 	offset_global = [tensor.tolist() for tensor in offset_global] # to list
	# 	print("here4")
	# 	# print('***offset_global', offset_global)
	# 	# print('table_size_list', table_size_list)
		
	# 	batch_num = len(offset_global)
	# 	print("here5")
	# 	# print('batch_num', batch_num)
	# 	batched_table_access = []
	# 	len_entries = []
	# 	for b in range(batch_num):
	# 		batched_table_access.append([b])
			
	# 	print("here6")
	# 	for i in range(len(embedding_table_gather_reduce_access)):
	# 		len_entries.append(len(embedding_table_gather_reduce_access[i][1]))
			
	# 		for j in embedding_table_gather_reduce_access[i][1]:
	# 			batched_table_access[i // len(table_size_list)].append((embedding_table_gather_reduce_access[i][0], j))
				
				
	# 	print("here7")
	# 	# print('batched_table_access', batched_table_access)
	# 	batched_table_access_list = []
	# 	# Modify the format
	# 	list_memory2  = sys.getsizeof(batched_table_access)
	# 	element_memory2 = sum(sys.getsizeof(elem) for elem in batched_table_access)
	# 	total_memory2 = list_memory2 + element_memory2
	# 	print("batched_table_access size: ",total_memory2)

	# 	for sublist in batched_table_access:
	# 		new_sublist = [[sublist[0]], sublist[1:]]
	# 		batched_table_access_list.append(new_sublist)
	# 		if(len(batched_table_access_list)%10000 == 0):
	# 			list_memory  = sys.getsizeof(batched_table_access_list)
	# 			element_memory = sum(sys.getsizeof(ele) for ele in batched_table_access_list)
	# 			total_memory = list_memory + element_memory
	# 			print("batched_table_access_list size: ",total_memory)
	# 	print("here8")

	# 	# print('batched_table_access_list', batched_table_access_list)
	# 	# batched_table_access_list == [[nth batch], [(kth table, kth entry), (kth table, kth entry), (kth table, kth entry)]]
	# 	# [[[0], [(0, 2), (0, 4), (0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (1, 4)]], [[1], [(2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 4), (3, 2), (3, 5)]]]
	# 	emb_table_pair = []
	# 	emb_table_pair = [(i, j) for i, size in enumerate(table_size_list) for j in range(size)]
	# 	print("here9")
	# 	# print(emb_table_pair)
	# 	# [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
	# 	mapped_dict = {idx: pair for idx, pair in enumerate(emb_table_pair)}
	# 	print("here10")
	# 	# print(mapped_dict)
	# 	# mapped_dict == {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (1, 0), 7: (1, 1), 8: (1, 2), 9: (1, 3), 10: (1, 4), 11: (1, 5)}

	# 	# print(len_entries)
	# 	#len_entries == [5, 3, 4, 5]
	# 	# offset_global == [[[0, 2], [0, 1]], [[0, 1], [0, 3]]]
	# 	entoffset = []

	# 	for i in range(len(offset_global)):
	# 		temp_result = []
	# 		for j in range(len(offset_global[i])):
	# 			temp_entry = offset_global[i][j] + [len_entries.pop(0)]
	# 			temp_result.append(temp_entry)
				
	# 		entoffset.append(temp_result)
	# 	print("here11")
		
	# 	# print(entoffset)

	# 	#entoffset == [[[0, 2, 5], [0, 1, 3]], [[0, 1, 4], [0, 3, 5]]]
	# 	# 5 entries wiht offset 0, 2 ......
	# 	# [[0,0,1,1,1,2,3,3], [0,1,1,1,2,2,2,3,3]]
	# 	entry_to_bag = [[] for _ in range(batch_num)]
	# 	print("here12")
	# 	for idx, group in enumerate(entoffset):
	# 		counter = 0
	# 		curr = 1
	# 		for i in range(len(group)):
	# 			for j in range(group[i][-1]):
	# 				if (curr < len(group[i])) and (len(group[i]) != 1):
	# 					if j >= group[i][curr]:
	# 						counter += 1
	# 						curr += 1
	# 				entry_to_bag[idx].append(counter)
	# 				print("here15")
	# 			counter += 1
	# 			curr = 1
	# 	print("here13")
	# 	# res == [[0, 0, 1, 1, 1, 2, 3, 3], [0, 1, 1, 1, 2, 2, 2, 3, 3]] # which entries belongs to which bag(res) (this example is two iteration)
	# 	# print('entry_to_bag: ', entry_to_bag)

	# 	gather_op_access = [[] for _ in range(batch_num)]
	# 	print("here14")
	# 	# reverse_mapped_dict = {v: k for k, v in mapped_dict.items()}
	# 	for idx, (_, accesses) in enumerate(batched_table_access_list):
	# 		mapped_indexes = []
	# 		for access in accesses:
	# 			for k, v in mapped_dict.items():
	# 				if v == access:
	# 					mapped_indexes.append(k)
						
	# 					break  
	# 		gather_op_access[idx].extend(mapped_indexes)
	# 	print("here15")

	# 	# print('gather_op_access: ', gather_op_access)
	# 	# gather_op_access == [[2, 4, 0, 2, 4, 7, 9, 10], [2, 3, 4, 5, 7, 8, 10, 8, 11]]
	# 	entry_to_bag_extend_to_mem_addr = [[val + len(mapped_dict) for val in sublist] for sublist in entry_to_bag]
	# 	print("here16")
	# 	# print('entry_to_bag_extend_to_mem_addr: ', entry_to_bag_extend_to_mem_addr)
	# 	# entry_to_bag_extend_to_mem_addr == [[12, 12, 13, 13, 13, 14, 15, 15], [12, 13, 13, 13, 14, 14, 14, 15, 15]]
	# 	mem_trace = [[] for _ in range(batch_num)]
	# 	print("here17")
	# 	# for inference gather reduce
	# 	for idx, access in enumerate(gather_op_access):
	# 		for i in range(len(access)):
	# 			mem_trace[idx].append((gather_op_access[idx][i], 'R'))
	# 			mem_trace[idx].append((entry_to_bag_extend_to_mem_addr[idx][i], 'R'))
	# 			mem_trace[idx].append((entry_to_bag_extend_to_mem_addr[idx][i], 'W'))
				
	# 		print(f"Size of mem_trace0: {sys.getsizeof(mem_trace)/(2**20)} MB")
	# 	print("here18")
	# 	# print('mem_trace: ', mem_trace) # for inference gather reduce ok
		
	# 	# size_of_the_reduced_embedding_vector_global == 4 for example
	# 	gradients_mem_addr = []
	# 	# gradients write back
	# 	for i in range(size_of_the_reduced_embedding_vector_global):
	# 		gradients_mem_addr.append(i + len(mapped_dict) + size_of_the_reduced_embedding_vector_global)
	# 	print("here19")
	# 	# print('gradients_mem_addr: ', gradients_mem_addr)
	# 	for trace in mem_trace:
	# 		for grad in gradients_mem_addr:
	# 			trace.append((grad, 'W'))
	# 	print("here20")
	# 	# print('mem_trace: ', mem_trace)
	# 	# gradients write back done
	# 	duplicated_grad_addr = [[] for _ in range(batch_num)]
	# 	print("here21")
	# 	for idx, access in enumerate(gather_op_access):
	# 		for i in range(len(access)):
	# 			duplicated_grad_addr[idx].append(i + max(gradients_mem_addr) + 1)
	# 	print("here22")
	# 	# print('duplicated_grad_addr: ', duplicated_grad_addr) 
	# 	# duplicated_grad_addr:  [[20, 21, 22, 23, 24, 25, 26, 27], [20, 21, 22, 23, 24, 25, 26, 27, 28]]

	# 	grad_to_duplicate_access = [[val + min(gradients_mem_addr) for val in sublist] for sublist in entry_to_bag]
	# 	print("here23")
	# 	# print('grad_to_duplicate_access: ', grad_to_duplicate_access)
	# 	# grad_to_duplicate_access:  [[16, 16, 17, 17, 17, 18, 19, 19], [16, 17, 17, 17, 18, 18, 18, 19, 19]]

	# 	# duplicate gradients operation
	# 	for idx, access in enumerate(grad_to_duplicate_access):
	# 		for i in range(len(access)):
	# 			mem_trace[idx].append((grad_to_duplicate_access[idx][i], 'R'))
	# 			mem_trace[idx].append((duplicated_grad_addr[idx][i], 'R'))
	# 			mem_trace[idx].append((duplicated_grad_addr[idx][i], 'W'))
				
	# 		print(f"Size of mem_trace1: {sys.getsizeof(mem_trace)/(2**20)} MB")
	# 	print("here24")
	# 	# print('mem_trace: ', mem_trace)
	# 	# coalescing gradients
	# 	# gather_op_access == [[2, 4, 0, 2, 4, 7, 9, 10], [2, 3, 4, 5, 7, 8, 10, 8, 11]]

	# 	coalesce_dst = []
	# 	for lst in gather_op_access:
	# 		# Find the unique elements and sort them
	# 		unique_sorted = sorted(set(lst)) # set(lst) removes duplicate values from lst, leaving only unique elements.
	# 		# Create a mapping from element to its rank
	# 		mapping = {value: index for index, value in enumerate(unique_sorted)}
	# 		# Remap the values in the list according to the mapping
	# 		remapped_list = [mapping[value] for value in lst] # remapped_list => indicate how big is the 
	# 		coalesce_dst.append(remapped_list)
	# 	print("here25")
	# 	# print('coalesce_dst: ', coalesce_dst)
	# 	# coalesce_dst:  [[1, 2, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 5, 7]]

	# 	coalesce_dst_addr = []
	# 	for idx, lst in enumerate(coalesce_dst):
	# 		max_addr = max(duplicated_grad_addr[idx])
	# 		coalesce_dst_addr.append([num + max_addr + 1 for num in lst])
	# 	print("here26")
	# 	# print('coalesce_dst_addr:', coalesce_dst_addr)
	# 	# coalesce_dst_addr: [[29, 30, 28, 29, 30, 31, 32, 33], [29, 30, 31, 32, 33, 34, 35, 34, 36]]
	# 	# duplicated_grad_addr:  [[20, 21, 22, 23, 24, 25, 26, 27], [20, 21, 22, 23, 24, 25, 26, 27, 28]]
		
	# 	#coalesce operation
	# 	for idx, access in enumerate(duplicated_grad_addr):
	# 		for i in range(len(access)):
	# 			mem_trace[idx].append((duplicated_grad_addr[idx][i], 'R'))
	# 			mem_trace[idx].append((coalesce_dst_addr[idx][i], 'R'))
	# 			mem_trace[idx].append((coalesce_dst_addr[idx][i], 'W'))
	# 		print(f"Size of mem_trace2: {sys.getsizeof(mem_trace)/(2**20)} MB")
	# 	print("here27")
	# 	# print('mem_trace: ', mem_trace)

	# 	write_back_to_table = [sorted(set(lst)) for lst in gather_op_access]
	# 	print("here28")
	# 	# print('write_back_to_table: ', write_back_to_table)
	# 	# write_back_to_table:  [[0, 2, 4, 7, 9, 10], [2, 3, 4, 5, 7, 8, 10, 11]]
	# 	coalesce_grad_ready_to_write_back = [sorted(set(lst)) for lst in coalesce_dst_addr]
	# 	print("here29")
	# 	# print('coalesce_grad_ready_to_write_back: ', coalesce_grad_ready_to_write_back)
	# 	# coalesce_grad_ready_to_write_back:  [[28, 29, 30, 31, 32, 33], [29, 30, 31, 32, 33, 34, 35, 36]]
		
	# 	#update emb table with coalesced gradients
	# 	for idx, access in enumerate(write_back_to_table):
	# 		for i in range(len(access)):
	# 			mem_trace[idx].append((coalesce_grad_ready_to_write_back[idx][i], 'R'))
	# 			mem_trace[idx].append((write_back_to_table[idx][i], 'R'))
	# 			mem_trace[idx].append((write_back_to_table[idx][i], 'W'))
	# 		print(f"Size of mem_trace3: {sys.getsizeof(mem_trace)/(2**20)} MB")
	# 	print("here30")
	# 	# print('standard mem_trace: ', mem_trace)

	# 	memory_needed = [max(lst) for lst in coalesce_dst_addr]
	# 	print(f"Size of memory_needed: {sys.getsizeof(memory_needed)/(2**20)} MB")
	# 	print("here31")
	# 	# print('memory_needed: ' , memory_needed)

	# 	add_op_count = []

	# 	for batch in mem_trace:
	# 		add_count = 0
	# 		for i in range(len(batch) - 1):
	# 			if (batch[i][1] == 'R') and (batch[i+1][1] == 'R'):
	# 				add_count += 1
	# 		add_op_count.append(add_count)
	# 	print('add_op_count_standard: ' , add_op_count)
	# 	print("here32")

	# 	return mem_trace, memory_needed

	# def memory_mapping(memory_trace, memory_needed, embedding_table_dimension_global):
	# 	base_address = 0x10000000  # base
	# 	address_shift_per_embedding_vector = 1 * embedding_table_dimension_global # the amount of address shift I need to take next vector

	# 	address_and_action_pair = [[] for _ in range(len(memory_trace))]
	# 	for idx, trace in enumerate(memory_trace):
	# 		all_address = [hex(base_address + address_shift_per_embedding_vector * i) for i in range(memory_needed[idx] + 1)]
	# 		# print("all_address", all_address)
	# 		for item in trace:
	# 			index, action = item
	# 			address = all_address[index]
	# 			address_and_action_pair[idx].append((address, action))
	# 		# address_and_action_pair[idx].append("STOP")
	# 	# print("address_and_action_pair", address_and_action_pair)

	# 	return address_and_action_pair



	# def write_output_to_txt(address_and_action_pair, file_name):
	# 	current_dir = os.getcwd()  
	# 	output_file_path = os.path.join(current_dir, file_name)  

	# 	with open(output_file_path, 'w') as file:
	# 		for sublist in address_and_action_pair:
	# 			for item in sublist:
	# 				if item[1] == 'W':  # Check if access type is 'W'
	# 					file.write(f"{item[0]} {item[1]}\n")  # Write with space
	# 				else:
	# 					file.write(f"{item[0]} {item[1]}\n")  # Write with space
	# 			file.write('STOP\n')  # Write STOP after each sublist
	
	# def access_count_compare(memtrace0, memtrace1):
	# 	res = [[] for _ in range(2)]
	# 	for i in range(len(memtrace0)):
	# 		res[0].append(len(memtrace0[i]))
	# 		res[1].append(len(memtrace1[i]))
	# 	# print(res)
	
	
# memory_trace, memory_needed = training_trace_standard(embedding_table_gather_reduce_access, embedding_table_len_global, size_of_the_reduced_embedding_vector_global, offset_global)
# address_and_action_pair = memory_mapping(memory_trace, memory_needed, embedding_table_dimension_global)
# #address_and_action_pair.append([(1,2),(3,4)])
# #print(address_and_action_pair)
# write_output_to_txt(address_and_action_pair, '003_5')
