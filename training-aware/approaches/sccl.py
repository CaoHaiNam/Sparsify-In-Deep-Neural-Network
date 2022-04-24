import sys, time, os
import math

import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
import time
import csv
from utils import cross_entropy, naive_lip
from sccl_layer import DynamicLinear, DynamicConv2D, _DynamicLayer, DynamicBatchNorm
import matplotlib.pyplot as plt
# import pygame
# from visualize import draw


class Appr(object):

	def __init__(self,model,args=None,thres=1e-3,lamb=0,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=10,optim='Adam'):
		self.model=model.to(device)

		self.nepochs = args.nepochs
		self.sbatch = args.batch_size
		self.lr = args.lr
		self.lr_min = lr/100
		self.lr_factor = lr_factor
		self.lr_patience = args.lr_patience
		self.clipgrad = clipgrad
		self.optim = args.optimizer
		self.thres = args.thres
		self.args = args
		
		self.lamb = args.lamb

		self.ce = torch.nn.CrossEntropyLoss()
		self.optimizer = self._get_optimizer()

		self.log_name = '{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, 
                                                                            args.lr, args.batch_size, args.nepochs, args.lamb)

		self.check_point = {'model':self.model, 'phase':3}

		if args.resume:
			try:
				self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
			except:
				pass

	def _get_optimizer(self,lr=None):
		if lr is None: lr=self.lr
		if self.optim == 'SGD':
			return torch.optim.SGD(self.model.parameters(), lr=lr,
						  weight_decay=0.0, momentum=0.9)
		if self.optim == 'Adam':
			return torch.optim.Adam(self.model.parameters(), lr=lr)

	def train(self, train_loader, valid_loader, prune_loader, ncla=0):

		if self.check_point['phase'] == 3:
			print('Training new task')

			# self.model.new_task()
			# self.model.expand(ncla, self.args.max_mul, self.max_params)
			# self.model.to(device)

				
			with open(f'../result_data/csv_data/{self.log_name}.csv', 'w', newline='') as csvfile:
				writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
				writer.writerow(['train loss', 'train acc', 'valid loss', 'valid acc'])
		else: 
			print('Retraining current task')


		self.train_phase(train_loader, valid_loader, prune_loader, True)
		if self.check_point['phase'] == 2:
			self.check_point = {'model':self.model, 'phase':3}
			torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

		# self.plot_neurons_importance()
		self.prune(prune_loader)


		self.check_point = {'model':self.model, 'phase':2, 'optimizer':self._get_optimizer(self.lr), 'epoch':-1, 'lr':self.lr, 'patience':self.lr_patience}
		torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

		self.train_phase(train_loader, valid_loader, prune_loader, False)

		self.check_point = {'model':self.model, 'phase':3}
		torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))

		

	def train_phase(self, train_loader, valid_loader, prune_loader, squeeze):


		if self.check_point['phase'] == 3:
			lr = self.lr
			patience = self.lr_patience
			self.optimizer = self._get_optimizer(lr)
			start_epoch = 0
			phase = 1
		else:
			self.model = self.check_point['model']
			lr = self.check_point['lr']
			patience = self.check_point['patience']
			self.optimizer = self.check_point['optimizer']
			phase = self.check_point['phase']
			start_epoch = self.check_point['epoch'] + 1

		print('number of neurons:', end=' ')
		for m in self.model.DM:
			print(m.out_features, end=' ')
		print()
		params = self.model.compute_model_size()
		print('num params', params)

		train_loss,train_acc=self.eval(prune_loader)
		print('| Train: loss={:.3f}, acc={:5.2f}% |'.format(train_loss,100*train_acc), end='')

		valid_loss,valid_acc=self.eval(valid_loader)
		print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc))

		if phase == 1:
			squeeze = True
		else:
			squeeze = False

		if squeeze:
			best_acc = train_acc
		else:
			best_acc = valid_acc
	
		try:
			for e in range(start_epoch, self.nepochs):
			# Train
				# if e % 10 == 0:
				# 	lip = naive_lip(self.model, 1000)
				# 	print('lip const:', lip)
				clock0=time.time()
				# self.optimizer = self._get_optimizer(lr)
				self.train_epoch(train_loader, squeeze)
			
				clock1=time.time()
				train_loss,train_acc=self.eval(prune_loader)
				clock2=time.time()
				print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
					e+1,1000*(clock1-clock0),
					1000*(clock2-clock1),train_loss,100*train_acc),end='')

				valid_loss,valid_acc=self.eval(valid_loader)
				print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
				
				# Adapt lr
				if squeeze:
					if train_acc >= best_acc:
						best_acc = train_acc
						self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'phase':phase, 'epoch':e, 'lr':lr, 'patience':patience}
						torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
						print(' *', end='')
						patience = self.lr_patience
					# else:
					# 	patience -= 1
					# 	if patience <= 0:
					# 		lr /= self.lr_factor
					# 		print(' lr={:.1e}'.format(lr), end='')
					# 		if lr < self.lr_min:
					# 			print()
					# 			break
								
					# 		patience = self.lr_patience
					# 		self.optimizer = self._get_optimizer(lr)

				else:
					if valid_acc > best_acc:
						best_acc = valid_acc
						self.check_point = {'model':self.model, 'optimizer':self.optimizer, 'phase':phase, 'epoch':e, 'lr':lr, 'patience':patience}
						torch.save(self.check_point,'../result_data/trained_model/{}.model'.format(self.log_name))
						patience = self.lr_patience
						print(' *', end='')
					else:
						patience -= 1
						if patience <= 0:
							lr /= self.lr_factor
							print(' lr={:.1e}'.format(lr), end='')
							if lr < self.lr_min:
								print()
								break
								
							patience = self.lr_patience
							self.optimizer = self._get_optimizer(lr)

				print()
				with open(f'../result_data/csv_data/{self.log_name}.csv', 'a', newline='') as csvfile:
					writer = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
					writer.writerow([train_loss, train_acc, valid_loss, valid_acc])

		except KeyboardInterrupt:
			print('KeyboardInterrupt')

		self.check_point = torch.load('../result_data/trained_model/{}.model'.format(self.log_name))
		self.model = self.check_point['model']
		self.model.to(device)

	def train_epoch(self, data_loader,squeeze=True):
		self.model.train()

		# Loop batches
		for images, targets in data_loader:
			
			images=images.to(device)
			targets=targets.to(device)

			# Forward current model
			outputs = self.model.forward(images, t=-1)

			loss = self.ce(outputs, targets)

			if squeeze:
				loss += self.model.group_lasso_reg() * self.lamb

			# Backward
			self.optimizer.zero_grad()
			loss.backward()
			# self.model.restrict_gradients()

			if self.optim == 'SGD':
				torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

			self.optimizer.step()


	def eval(self,data_loader,masks=None):
		total_loss=0
		total_acc=0
		total_num=0
		self.model.eval()

		# Loop batches
		for images, targets in data_loader:
			images=images.to(device)
			targets=targets.to(device)
				
			# Forward
			if masks is not None:
				outputs = self.model.forward_mask(images, masks, t=-1)
			else:
				outputs = self.model.forward(images, t=-1)
					
			loss=self.ce(outputs,targets)
			values,indices=outputs.max(1)
			hits=(indices==targets).float()

			total_loss+=loss.data.cpu().numpy()*len(targets)
			total_acc+=hits.sum().data.cpu().numpy()
			total_num+=len(targets)

		return total_loss/total_num,total_acc/total_num

	def prune(self, data_loader, thres=0.0):

		loss,acc=self.eval(data_loader,None)
		loss, acc = round(loss, 3), round(acc, 3)
		print('Pre Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))
		# pre_prune_acc = acc
		pre_prune_loss = loss
		prune_ratio = np.ones(len(self.model.DM)-1)
		step = 0
		masks = [None for m in self.model.DM[:-1]]

		# Dynamic expansion
		while sum(prune_ratio) > 0.0:
			t1 = time.time()
			fig, axs = plt.subplots(1, len(self.model.DM)-1, figsize=(3*len(self.model.DM)-3, 2))
			# fig1, axs1 = plt.subplots(1, 3, figsize=(9, 2))
			mask_in = torch.ones(self.model.DM[0].in_features).bool().cuda()
			print('Pruning ratio:', end=' ')
			for i in range(0, len(self.model.DM)-1):
				m = self.model.DM[i]
				mask_out = torch.ones(m.shape_out[-1]).bool().to(device)
				# remove neurons
				m.squeeze(mask_in, mask_out)
				mask_in = torch.ones(m.in_features).bool().cuda()
				norm = m.norm_in() * m.bn_norm()
				if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
					# norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
					# 											m.shape_out[-1]-m.shape_out[-2], 
					# 											self.model.smid, self.model.smid))

					norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
																m.shape_out[-1], 
																self.model.smid, self.model.smid))
				else:
					norm *= self.model.DM[i+1].norm_out()

				norm = norm[m.shape_out[-2]:]
				# m.omega = 1/norm
				low, high = 0, norm.shape[0]

				#visualize
				# if i in [4, 5, 6]:
				# 	axs1[i-4].hist(norm.detach().cpu().numpy(), bins=100)
				# 	axs1[i-4].set_title(f'layer {i+1}')

				axs[i].hist(norm.detach().cpu().numpy(), bins=100)
				axs[i].set_title(f'layer {i+1}')


				if norm.shape[0] != 0:
					values, indices = norm.sort(descending=True)
					while True:
						k = (high+low)//2
						# if k == norm.shape[0]:
						# 	masks[i][m.shape_out[-2]:] = True
						# else:
						# Select top-k biggest norm
						mask_out[m.shape_out[-2]:] = (norm>values[k])
						# mask_out = (norm>values[k])
						masks[i] = mask_out
						loss, acc = self.eval(data_loader, masks)
						loss, acc = round(loss, 3), round(acc, 3)
						# post_prune_acc = acc
						post_prune_loss = loss
						if post_prune_loss <= pre_prune_loss:
						# if (pre_prune_acc - post_prune_acc) <= thres:
							# k is satisfy, try smaller k
							high = k
						else:
							# k is not satisfy, try bigger k
							low = k

						if k == (high+low)//2:
							break

				if high == norm.shape[0]:
					# not found any k satisfy, keep all neurons
					mask_out[m.shape_out[-2]:] = True
					# mask_out = True
				else:
					# found k = high is the smallest k satisfy
					mask_out[m.shape_out[-2]:] = (norm>values[high])
					# mask_out = (norm>values[high])

				masks[i] = None
				# remove neurons 
				m.squeeze(mask_in, mask_out)

				if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
					mask_in = mask_out.view(-1,1,1).expand(mask_out.size(0),self.model.smid,self.model.smid).contiguous().view(-1)
				else:
					mask_in = mask_out

				mask_count = int(sum(mask_out[m.shape_out[-2]:]))
				total_count = int(np.prod(mask_out[m.shape_out[-2]:].shape))
				if total_count == 0:
					prune_ratio[i] = 0.0
				else:
					prune_ratio[i] = 1.0 - mask_count/total_count
				print('{:.3f}'.format(prune_ratio[i]), end=' ')

			mask_out = torch.ones(self.model.DM[-1].out_features).bool().cuda()
			self.model.DM[-1].squeeze(mask_in, mask_out)
			print('| Time={:5.1f}ms'.format((time.time()-t1)*1000))
			# self.model.squeeze(masks)
			# fig.savefig(f'../result_data/images/{self.log_name}_task{t}_step_{step}.pdf', bbox_inches='tight')
			# fig1.savefig(f'../result_data/images/{self.file_name}_task{t}_step_{step}_half.pdf', bbox_inches='tight')
			# plt.show()
			step += 1
			# break
		loss,acc=self.eval(data_loader,None)
		print('Post Prune: loss={:.3f}, acc={:5.2f}% |'.format(loss,100*acc))

		print('number of neurons:', end=' ')
		for m in self.model.DM:
			print(m.out_features, end=' ')
		print()
		params = self.model.compute_model_size()
		print('num params', params)


	def plot_neurons_importance(self):
		fig, axs = plt.subplots(1, len(self.model.DM), figsize=(3*len(self.model.DM)-3, 2))
		for i in range(0, len(self.model.DM)-1):
			m = self.model.DM[i]
			norm = m.norm_in() * m.bn_norm()
			if isinstance(m, DynamicConv2D) and isinstance(self.model.DM[i+1], DynamicLinear):
				# norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
				# 											m.shape_out[-1]-m.shape_out[-2], 
				# 											self.model.smid, self.model.smid))
				norm *= self.model.DM[i+1].norm_out(size=(self.model.DM[i+1].shape_out[-1], 
															m.shape_out[-1], 
															self.model.smid, self.model.smid))
			else:
				norm *= self.model.DM[i+1].norm_out()


			axs[i].hist(norm.detach().cpu().numpy(), bins=100)
			axs[i].set_title(f'layer {i+1}')

		norm = self.model.DM[-1].norm_in()
		print(norm)
		axs[-1].hist(norm.detach().cpu().numpy(), bins=100)
		plt.show()

