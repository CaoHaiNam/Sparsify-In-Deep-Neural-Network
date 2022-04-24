import sys,time,os
import numpy as np
import torch
from copy import deepcopy
import utils
from utils import *
sys.path.append('..')
from arguments import get_args
import torch.nn.functional as F
import torch.nn as nn

args = get_args()

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=10,clipgrad=100, args=None, log_name=None, split=False):
        self.model=model

        self.log_name = '{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}'.format(args.date, args.experiment, args.approach, args.seed, 
                                                                            args.lr, args.batch_size, args.nepochs)
        self.logger = utils.logger(file_name=self.log_name, resume=False, path='../result_data/csv_data/', data_format='csv')

        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.lr_min = self.lr/100
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        
        if args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.9)
        if args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
#         return torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    def train(self, train_loader, valid_loader):
        best_acc = -np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                            
                train_loss,train_acc = self.train_epoch(train_loader)
                
                clock1=time.time()
                # train_loss,train_acc=self.eval(train_loader)
                valid_loss,valid_acc=self.eval(valid_loader)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                    e+1, 1000*(clock1-clock0), 1000*(clock2-clock1),train_loss,100*train_acc),end='')
                # Valid
                # valid_loss,valid_acc=self.eval(valid_loader)
                print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
                
                #save log for current task & old tasks at every epoch
                self.logger.add(epoch=e, train_loss=train_loss, train_acc=train_acc, valid_loss=valid_loss, valid_acc=valid_acc)
                
                # Adapt lr
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = utils.get_model(self.model)
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
                            if args.conv_net:
                                pass
    #                             break
                        patience = self.lr_patience
                        self.optimizer = self._get_optimizer(lr)
                print()
                
        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        # Restore best
        utils.set_model_(self.model, best_model)

        self.logger.save()
        
        # Update old
        torch.save(self.model, '../result_data/trained_model/' + self.log_name + '.model')

        return

    def train_epoch(self,data_loader):
        self.model.train()

        total_loss=0
        total_acc=0
        total_num=0

        # Loop batches
        for images, targets in data_loader:
            images = images.cuda()
            targets = targets.cuda()
            # Forward current model
            outputs = self.model.forward(images)
            loss=self.ce(outputs,targets)

            _,pred=outputs.max(1)
            hits=(pred==targets).float()

            total_loss+=loss.data.cpu().numpy()*len(targets)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

        return total_loss/total_num,total_acc/total_num

    def eval(self,data_loader):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()


        # Loop batches
        for images, targets in data_loader:
            
            images = images.cuda()
            targets = targets.cuda()
            # Forward
            outputs = self.model.forward(images)
                
            loss=self.ce(outputs,targets)
            _,pred=outputs.max(1)
            hits=(pred==targets).float()

            # Log
#             total_loss+=loss.data.cpu().numpy()[0]*len(b)
#             total_acc+=hits.sum().data.cpu().numpy()[0]
            total_loss+=loss.data.cpu().numpy()*len(targets)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(targets)

        return total_loss/total_num,total_acc/total_num


    def prune(self, model):
        for i, m in enumerate(model.PM):
            N = m.weight.shape[0]
            W = m.weight.view(N, -1)
            W = W / W.norm(2, dim=1)
            W = torch.cat((W, m.bias.view(1, -1)), dim=1)
            s = ((W.view(N, 1, -1).expand(N, N, -1) - W.view(1, N, -1).expand(N, N, -1))**2).sum(-1) / 
                ((W.view(N, 1, -1).expand(N, N, -1) + W.view(1, N, -1).expand(N, N, -1))**2).sum(-1)
            if len(model.PM[i+1].weight.shape) == 4:
                s *= (model.PM[i+1].weight**2).sum((0, 2, 3))
            else:
                s *= (model.PM[i+1].weight**2).sum(0)

            v, i = s.view(-1).sort(descending=True)

            mask = torch.ones(N).bool().cuda()
            while True:
                k = (high+low)//2
                # Select top-k biggest norm
                mask = (norm>values[k])
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
