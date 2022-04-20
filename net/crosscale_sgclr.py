import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
from feeder.tools import reduce2part

class CrosScale_SGCLR(nn.Module):
    """ Referring to the code of MOCO, https://arxiv.org/abs/1911.05722 """

    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, **kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """

        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain

        if not self.pretrain:
            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_scale = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=num_class,
                                                 dropout=dropout, graph_args={'layout':'coarse-ntu-rgb+d','strategy':'spatial'},
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature

            self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=feature_dim,
                                          dropout=dropout, graph_args={'layout':'ntu-rgb+d','strategy':'uniform'},
                                          edge_importance_weighting=edge_importance_weighting,
                                          **kwargs)
            self.encoder_q_scale = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args={'layout':'coarse-ntu-rgb+d','strategy':'spatial'},
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)
            self.encoder_k_scale = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                                 hidden_dim=hidden_dim, num_class=feature_dim,
                                                 dropout=dropout, graph_args={'layout':'coarse-ntu-rgb+d','strategy':'uniform'},
                                                 edge_importance_weighting=edge_importance_weighting,
                                                 **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)
                self.encoder_q_scale.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_q.fc)
                self.encoder_k_scale.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                         nn.ReLU(),
                                                         self.encoder_k.fc)

            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)    # initialize
                param_k.requires_grad = False       # not update by gradient
            for param_q, param_k in zip(self.encoder_q_scale.parameters(), self.encoder_k_scale.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            # create the queue
            self.register_buffer("queue", torch.randn(feature_dim, self.K))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.register_buffer("queue_scale", torch.randn(feature_dim, self.K))
            self.queue_scale = F.normalize(self.queue_scale, dim=0)
            self.register_buffer("queue_ptr_scale", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_scale(self):
        for param_q, param_k in zip(self.encoder_q_scale.parameters(), self.encoder_k_scale.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
    
    @torch.no_grad()
    def _dequeue_and_enqueue_scale(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_scale)
        gpu_index = keys.device.index
        self.queue_scale[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K
        self.queue_ptr_scale[0] = (self.queue_ptr_scale[0] + batch_size) % self.K

    def forward(self, im_q, im_k=None, view='all', cross=False, topk=1, context=True): 
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """
        if cross:
            return self.cross_training(im_q, im_k, topk, context)

        # im_q_scale=reduce2part(im_q)
        im_q_scale = torch.zeros_like(im_q)
        im_q_scale[:, :, :, 0, :] = (im_q[:, :, :, 0, :] + im_q[:, :, :, 1, :]) / 2
        im_q_scale[:, :, :, 1, :] = (im_q[:, :, :, 2, :] + im_q[:, :, :, 3, :] + im_q[:, :, :, 20, :]) / 3
        im_q_scale[:, :, :, 2, :] = (im_q[:, :, :, 4, :] + im_q[:, :, :, 5, :]) / 2
        im_q_scale[:, :, :, 3, :] = (im_q[:, :, :, 6, :] + im_q[:, :, :, 7, :] + im_q[:, :, :, 21, :] + im_q[:, :, :,22, :]) / 4
        im_q_scale[:, :, :, 4, :] = (im_q[:, :, :, 8, :] + im_q[:, :, :, 9, :]) / 2
        im_q_scale[:, :, :, 5, :] = (im_q[:, :, :, 10, :] + im_q[:, :, :, 11, :] + im_q[:, :, :, 23, :] + im_q[:, :, :,24, :]) / 4
        im_q_scale[:, :, :, 6, :] = (im_q[:, :, :, 12, :] + im_q[:, :, :, 13, :]) / 2
        im_q_scale[:, :, :, 7, :] = (im_q[:, :, :, 14, :] + im_q[:, :, :, 15, :]) / 2
        im_q_scale[:, :, :, 8, :] = (im_q[:, :, :, 16, :] + im_q[:, :, :, 17, :]) / 2
        im_q_scale[:, :, :, 9, :] = (im_q[:, :, :, 18, :] + im_q[:, :, :, 19, :]) / 2
        im_q_scale = im_q_scale[:, :, :, :10, :]

        if not self.pretrain:
            if view == 'joint':
                return self.encoder_q(im_q)
            elif view == 'scale':
                return self.encoder_q_scale(im_q_scale)
            elif view == 'all':
                return (self.encoder_q(im_q) + self.encoder_q_motion(im_q_scale)) / 2.
            else:
                raise ValueError

        # im_k_scale=reduce2part(im_k)
        im_k_scale = torch.zeros_like(im_k)
        im_k_scale[:, :, :, 0, :] = (im_k[:, :, :, 0, :] + im_k[:, :, :, 1, :]) / 2
        im_k_scale[:, :, :, 1, :] = (im_k[:, :, :, 2, :] + im_k[:, :, :, 3, :] + im_k[:, :, :, 20, :]) / 3
        im_k_scale[:, :, :, 2, :] = (im_k[:, :, :, 4, :] + im_k[:, :, :, 5, :]) / 2
        im_k_scale[:, :, :, 3, :] = (im_k[:, :, :, 6, :] + im_k[:, :, :, 7, :] + im_k[:, :, :, 21, :] + im_k[:, :, :,22, :]) / 4
        im_k_scale[:, :, :, 4, :] = (im_k[:, :, :, 8, :] + im_k[:, :, :, 9, :]) / 2
        im_k_scale[:, :, :, 5, :] = (im_k[:, :, :, 10, :] + im_k[:, :, :, 11, :] + im_k[:, :, :, 24, :] + im_k[:, :, :,23, :]) / 4
        im_k_scale[:, :, :, 6, :] = (im_k[:, :, :, 12, :] + im_k[:, :, :, 13, :]) / 2
        im_k_scale[:, :, :, 7, :] = (im_k[:, :, :, 14, :] + im_k[:, :, :, 15, :]) / 2
        im_k_scale[:, :, :, 8, :] = (im_k[:, :, :, 16, :] + im_k[:, :, :, 17, :]) / 2
        im_k_scale[:, :, :, 9, :] = (im_k[:, :, :, 18, :] + im_k[:, :, :, 19, :]) / 2
        im_k_scale = im_k_scale[:, :, :, :10, :]

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        q_scale = self.encoder_q_scale(im_q_scale)
        q_scale = F.normalize(q_scale, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            self._momentum_update_key_encoder_scale()

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

            k_scale = self.encoder_k_scale(im_k_scale)
            k_scale = F.normalize(k_scale, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_scale = torch.einsum('nc,nc->n', [q_scale, k_scale]).unsqueeze(-1)
        l_neg_scale = torch.einsum('nc,ck->nk', [q_scale, self.queue_scale.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_scale = torch.cat([l_pos_scale, l_neg_scale], dim=1)

        # apply temperature
        logits /= self.T
        logits_scale /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_scale(k_scale)

        return logits, logits_scale, labels

    def cross_training(self, im_q, im_k, topk=1, context=True):
        # im_q_scale=reduce2part(im_q)
        im_q_scale = torch.zeros_like(im_q)
        im_q_scale[:, :, :, 0, :] = (im_q[:, :, :, 0, :] + im_q[:, :, :, 1, :]) / 2
        im_q_scale[:, :, :, 1, :] = (im_q[:, :, :, 2, :] + im_q[:, :, :, 3, :] + im_q[:, :, :, 20, :]) / 3
        im_q_scale[:, :, :, 2, :] = (im_q[:, :, :, 4, :] + im_q[:, :, :, 5, :]) / 2
        im_q_scale[:, :, :, 3, :] = (im_q[:, :, :, 6, :] + im_q[:, :, :, 7, :] + im_q[:, :, :, 21, :] + im_q[:, :, :,22, :]) / 4
        im_q_scale[:, :, :, 4, :] = (im_q[:, :, :, 8, :] + im_q[:, :, :, 9, :]) / 2
        im_q_scale[:, :, :, 5, :] = (im_q[:, :, :, 10, :] + im_q[:, :, :, 11, :] + im_q[:, :, :, 24, :] + im_q[:, :, :,23, :]) / 4
        im_q_scale[:, :, :, 6, :] = (im_q[:, :, :, 12, :] + im_q[:, :, :, 13, :]) / 2
        im_q_scale[:, :, :, 7, :] = (im_q[:, :, :, 14, :] + im_q[:, :, :, 15, :]) / 2
        im_q_scale[:, :, :, 8, :] = (im_q[:, :, :, 16, :] + im_q[:, :, :, 17, :]) / 2
        im_q_scale[:, :, :, 9, :] = (im_q[:, :, :, 18, :] + im_q[:, :, :, 19, :]) / 2
        im_q_scale = im_q_scale[:, :, :, :10, :]

        # im_k_scale=reduce2part(im_k)
        im_k_scale = torch.zeros_like(im_k)
        im_k_scale[:, :, :, 0, :] = (im_k[:, :, :, 0, :] + im_k[:, :, :, 1, :]) / 2
        im_k_scale[:, :, :, 1, :] = (im_k[:, :, :, 2, :] + im_k[:, :, :, 3, :] + im_k[:, :, :, 20, :]) / 3
        im_k_scale[:, :, :, 2, :] = (im_k[:, :, :, 4, :] + im_k[:, :, :, 5, :]) / 2
        im_k_scale[:, :, :, 3, :] = (im_k[:, :, :, 6, :] + im_k[:, :, :, 7, :] + im_k[:, :, :, 21, :] + im_k[:, :, :,22, :]) / 4
        im_k_scale[:, :, :, 4, :] = (im_k[:, :, :, 8, :] + im_k[:, :, :, 9, :]) / 2
        im_k_scale[:, :, :, 5, :] = (im_k[:, :, :, 10, :] + im_k[:, :, :, 11, :] + im_k[:, :, :, 24, :] + im_k[:, :, :,23, :]) / 4
        im_k_scale[:, :, :, 6, :] = (im_k[:, :, :, 12, :] + im_k[:, :, :, 13, :]) / 2
        im_k_scale[:, :, :, 7, :] = (im_k[:, :, :, 14, :] + im_k[:, :, :, 15, :]) / 2
        im_k_scale[:, :, :, 8, :] = (im_k[:, :, :, 16, :] + im_k[:, :, :, 17, :]) / 2
        im_k_scale[:, :, :, 9, :] = (im_k[:, :, :, 18, :] + im_k[:, :, :, 19, :]) / 2
        im_k_scale = im_k_scale[:, :, :, :10, :]

        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        q_scale = self.encoder_q_scale(im_q_scale)
        q_scale = F.normalize(q_scale, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder_scale()

            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            
            k_scale = self.encoder_k_scale(im_k_scale)
            k_scale = F.normalize(k_scale, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_scale = torch.einsum('nc,nc->n', [q_scale, k_scale]).unsqueeze(-1)
        l_neg_scale = torch.einsum('nc,ck->nk', [q_scale, self.queue_scale.clone().detach()])
        
        if context:
            l_context = torch.einsum('nk,nk->nk', [l_neg, l_neg_scale])
            logits = torch.cat([l_pos, l_neg, l_context], dim=1)
            logits_scale = torch.cat([l_pos_scale, l_neg_scale, l_context], dim=1)
        else:
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits_scale = torch.cat([l_pos_scale, l_neg_scale], dim=1)
        
        logits /= self.T
        logits_scale /= self.T

        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_scale = torch.topk(l_neg_scale, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)
        topk_onehot_scale = torch.zeros_like(l_neg_scale)

        topk_onehot.scatter_(1, topkdix_scale, 1)
        topk_onehot_scale.scatter_(1, topkdix, 1)

        if context:
            pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot, topk_onehot], dim=1)
            pos_mask_scale = torch.cat([torch.ones(topk_onehot_scale.size(0), 1).cuda(), topk_onehot_scale, topk_onehot_scale], dim=1)
        else:
            pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)
            pos_mask_scale = torch.cat([torch.ones(topk_onehot_scale.size(0), 1).cuda(), topk_onehot_scale], dim=1)

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_scale(k_scale)

        return logits, logits_scale, pos_mask, pos_mask_scale
