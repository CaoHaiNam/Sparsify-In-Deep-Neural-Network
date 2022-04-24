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
from networks.architecture_learning_net import tsReLU
args = get_args()

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=10,clipgrad=100, args=None, log_name=None, split=False):
        self.model=model

        self.log_name = '{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}_lamb_w_{}_lamb_d_{}_smax_{}'.format(args.date, args.experiment, args.approach, args.seed, 
                                                                            args.lr, args.batch_size, args.nepochs, args.lamb_w, args.lamb_d, args.smax)
        self.logger = utils.logger(file_name=self.log_name, resume=False, path='../result_data/csv_data/', data_format='csv')

        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.lr_min = self.lr/100
        self.lr_factor = lr_factor
        self.lr_patience = args.lr_patience
        self.clipgrad = clipgrad
        self.lamb_w = args.lamb_w
        self.lamb_d = args.lamb_d
        self.smax = args.smax 

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
        # best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        # s = 1
        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                            
                train_loss,train_acc = self.train_epoch(train_loader)
                # s *= self.smax ** (1./self.nepochs)
                
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
                    # best_model = utils.get_model(self.model)
                    torch.save(self.model, '../result_data/trained_model/' + self.log_name + '.model')
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
        # utils.set_model_(self.model, best_model)

        self.logger.save()
        
        # Update old
        # torch.save(self.model, '../result_data/trained_model/' + self.log_name + '.model')
        self.model = torch.load('../result_data/trained_model/' + self.log_name + '.model')
        self.prune(self.model)
        return

    def train_epoch(self,data_loader):
        self.model.train()

        total_loss=0
        total_acc=0
        total_num=0

        # Loop batches
        i = 0
        for images, targets in data_loader:
            images = images.cuda()
            targets = targets.cuda()

            s=(self.smax-1/self.smax)*i*len(targets)/len(data_loader)+1/self.smax
            # Forward current model
            outputs = self.model.forward(images, s)
            loss=self.criterion(outputs, targets, s)

            _,pred=outputs.max(1)
            hits=(pred==targets).float()

            total_loss+=loss.data.cpu().numpy()*len(targets)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            for n,p in self.model.named_parameters():
                if n == 'w' or n == 'd':
                    num=torch.cosh(torch.clamp(s*p.data,-50,50))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            if args.optimizer == 'SGD' or args.optimizer == 'SGD_momentum_decay':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()

            for n,p in self.model.named_parameters():
                if n == 'w' or n == 'd':
                    p.data=torch.clamp(p.data,-6,6)

            i+=1

        for i, m in enumerate(self.model.layers):
            if isinstance(m, tsReLU):
                # print((m.w>=0))
                # print(m.d)
                # print('w:{}-d:{}'.format(m.gate(self.smax*m.w).sum(), m.gate(self.smax*m.d).sum()), end=' ')
                print('w:{}-d:{}'.format((m.w>=0).sum().item(), (m.d>=0).sum().item()), end=' ')

        print()
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
            outputs = self.model.forward(images, self.smax)
                
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

    # def criterion(self, outputs, targets, s):
    #     reg = 0
    #     num = 0
    #     modules = list(self.model.modules())

    #     for i, m in enumerate(modules):
    #         if isinstance(m, tsReLU):
    #             # if isinstance(modules[i+1], nn.MaxPool2d):
    #             #     reg += modules[i-1].weight.numel() * (self.lamb_w*m.w.sum())
    #             # else:
    #             #     reg += modules[i-1].weight.numel() * (self.lamb_w*m.w.sum() - self.lamb_d*m.d.sum())

    #             reg += 2 * (self.lamb_w*(m.w*(1-m.w)).sum() + self.lamb_d*(m.d*(1-m.d)).sum())
    #             num += modules[i-1].weight.numel()

    #     return self.ce(outputs, targets) + reg.sum()


    def criterion(self, outputs, targets, s):
        reg = 0
        num = 0

        for i, m in enumerate(self.model.PM):
            if isinstance(m, tsReLU):
                # if isinstance(self.model.PM[i-1], nn.Conv2d):
                #     reg += self.model.PM[i-1].weight.numel() * (self.lamb_w*m.gate(s*m.w).sum())
                #     # print(' no d', end=' ')
                # else:
                reg += self.model.PM[i-1].weight.numel() * (self.lamb_w*m.gate(s*m.w).sum()*(m.d<=0).sum() - self.lamb_d*m.gate(s*m.d).sum())
                    # print(' d', end=' ')

                num += self.model.PM[i-1].weight.numel()
        # print()
        return self.ce(outputs, targets) + reg / num

    def prune(self, model):

        new_layers = nn.ModuleList([])
        temp_layers = []
        last_layer = None
        merge = False

        if isinstance(model.layers[0], nn.Linear):
            mid = True
            mask_in = torch.ones(model.layers[0].in_features).cuda().bool()
        else:
            mid = False
            mask_in = torch.ones(model.layers[0].in_channels).cuda().bool()

        for i, m in enumerate(model.layers[:-1]):

            if isinstance(m ,tsReLU):

                if isinstance(model.layers[i-1], nn.Linear) and not mid:
                    mid = True
                    mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0),model.smid,model.smid).contiguous().view(-1)

                mask_out = m.w >= 0
                m.w.data = m.w.data[mask_out]
                model.layers[i-1].weight.data = model.layers[i-1].weight.data[mask_out][:, mask_in]
                model.layers[i-1].bias.data = model.layers[i-1].bias.data[mask_out]

                m.out_features, m.in_features = model.layers[i-1].weight.shape[0], model.layers[i-1].weight.shape[1]
                mask_in = mask_out

        model.layers[-1].weight.data = model.layers[-1].weight.data[:, mask_in]

        for i, m in enumerate(model.layers):

            if not isinstance(m, nn.Linear) and not isinstance(m, tsReLU):
                new_layers.append(m)
                continue

            if isinstance(m, nn.Linear):
                if merge:
                    new_layer = nn.Linear(last_layer.in_features, m.out_features)
                    w_new = torch.matmul(m.weight, last_layer.weight)
                    b_new = torch.matmul(m.weight, last_layer.bias.view(-1, 1)) + m.bias.view(-1, 1)
                    new_layer.weight.data = w_new
                    new_layer.bias.data = b_new.view(-1)
                    last_layer = new_layer
                    temp_layers.append(new_layer)
                else:
                    last_layer = m
                    temp_layers.append(m)
            else:
                temp_layers.append(m)

            if isinstance(m ,tsReLU):
                if m.d >= 0 and isinstance(model.layers[i-1], nn.Linear):
                    merge = True
                    temp_layers = []
                else:
                    merge = False
                    new_layers += temp_layers
                    temp_layers = []

        new_layers += temp_layers
        model.layers = new_layers
        print(print_model_report(model))
    