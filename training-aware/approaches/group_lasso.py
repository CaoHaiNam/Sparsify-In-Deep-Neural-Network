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

from networks.vbd_vgg import VBD_Layer

args = get_args()

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-6,lr_factor=3,lr_patience=10,clipgrad=100, args=None, log_name=None, split=False):
        self.model=model

        self.log_name = '{}_{}_{}_{}_lr_{}_batch_{}_epoch_{}_lamb_{}'.format(args.date, args.experiment, args.approach, args.seed, 
                                                                            args.lr, args.batch_size, args.nepochs, args.lamb)
        self.logger = utils.logger(file_name=self.log_name, resume=False, path='../result_data/csv_data/', data_format='csv')

        self.nepochs = args.nepochs
        self.sbatch = args.batch_size
        self.lr = args.lr
        self.lr_min = self.lr/100
        self.lr_factor = lr_factor
        self.lr_patience = args.lr_patience
        self.lamb = args.lamb
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
        # best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        
        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                            
                train_loss,train_acc = self.train_epoch(train_loader,lr)
                
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
        self.model = torch.load('../result_data/trained_model/' + self.log_name + '.model')
        self.prune(self.model)
        return

    def train_epoch(self,data_loader,lr):
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

        self.proxy_grad_descent(lr)

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


    def proxy_grad_descent(self, lr):
        num_zeros = 0
        num = 0
        with torch.no_grad():
            # modules = list(self.model.modules())
            for i, module in enumerate(self.model.PM[:-1]):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue
                
                mu = self.lamb
                
                weight = module.weight
                bias = module.bias
                
                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1,2,3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm**2 + bias**2).pow(1/2)

                aux = F.threshold(norm - mu*lr*weight.numel(), 0, 0, False)
                alpha = aux/(aux + mu*lr*weight.numel())
                coeff = alpha

                if len(weight.size()) > 2:
                    weight.data = weight.data * coeff.view(-1, 1, 1, 1)
                else:
                    weight.data = weight.data * coeff.view(-1, 1) 

                bias.data = bias.data * coeff

                # if len(self.model.PM[i+1].weight.size()) > 2:
                #     self.model.PM[i+1].weight.data = self.model.PM[i+1].weight.data * coeff.view(1, -1, 1, 1)
                # else:
                #     self.model.PM[i+1].weight.data = self.model.PM[i+1].weight.data * coeff.view(1, -1)

                num_zeros += (weight.data == 0).float().sum() + (bias.data == 0).float().sum()
                num += weight.data.numel() + bias.data.numel()
                print((coeff == 0).float().sum().item(), end=' ')

        print('Sparsity: {:5.2f}%'.format(100.0*num_zeros/num))

        return


    def prune(self, model):

        if isinstance(model.PM[0], nn.Linear):
            mid = True
            mask_in = torch.ones(model.PM[0].in_features).cuda().bool()
        else:
            mid = False
            mask_in = torch.ones(model.PM[0].in_channels).cuda().bool()


        for i, m in enumerate(model.PM[:-1]):


            if isinstance(m, nn.Linear) and not mid:
                mid = True
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0),model.smid,model.smid).contiguous().view(-1)

            mu = self.lamb  
            weight = m.weight
            bias = m.bias
                
            if len(weight.size()) > 2:
                norm = weight.norm(2, dim=(1,2,3))
            else:
                norm = weight.norm(2, dim=(1))
            norm = (norm**2 + bias**2).pow(1/2)

            mask_out = (norm != 0)
            weight.data = weight.data[mask_out][:, mask_in]
            bias.data = bias.data[mask_out]

            m.out_features, m.in_features = weight.shape[0], weight.shape[1]
            mask_in = mask_out

        model.PM[-1].weight.data = model.PM[-1].weight.data[:, mask_in]

        print(print_model_report(model))

