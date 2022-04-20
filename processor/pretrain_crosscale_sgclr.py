#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class CrosScale_SGCLR_Processor(PT_Processor):
    """
        Processor for CrosSCLR Pretraining.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_scale_value = []

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.cross_epoch:
                output, output_scale, target = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
                loss_scale = self.loss(output_scale, target)

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_scale'] = loss_scale.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_scale_value.append(self.iter_info['loss_scale'])
                loss = loss + loss_scale
            else:
                output, output_scale, mask, mask_scale = self.model(data1, data2, cross=True, topk=self.arg.topk, context=self.arg.context)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = - (F.log_softmax(output, dim=1) * mask).sum(1) / mask.sum(1)
                loss_scale = - (F.log_softmax(output_scale, dim=1) * mask_scale).sum(1) / mask_scale.sum(1)
                loss = loss.mean()
                loss_scale = loss_scale.mean()

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_scale'] = loss_scale.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_scale_value.append(self.iter_info['loss_scale'])
                loss = loss + loss_scale

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            self.train_writer.add_scalar('batch_loss_scale', self.iter_info['loss_scale'], self.global_step)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_scale']= np.mean(loss_scale_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_scale', self.epoch_info['train_mean_loss_scale'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-scale training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-scale training')
        # endregion yapf: enable

        return parser
