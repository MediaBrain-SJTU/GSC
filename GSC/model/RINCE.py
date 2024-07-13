import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from model.SGRAF import SGRAF
from sklearn.mixture import GaussianMixture

import numpy as np
from scipy.spatial.distance import cdist
from utils import AverageMeter

class RINCE(nn.Module):
    def __init__(self, opt):
        super(RINCE, self).__init__()
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.similarity_model = SGRAF(opt)
        self.params = list(self.similarity_model.params)
        self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        self.step = 0

    def state_dict(self):
        return self.similarity_model.state_dict()

    def load_state_dict(self, state_dict):
        self.similarity_model.load_state_dict(state_dict)

    def train_start(self):
        """switch to train mode"""
        self.similarity_model.train_start()

    def val_start(self):
        """switch to valuate mode"""
        self.similarity_model.val_start()

    def info_nce_loss(self, similarity_matrix, temp=None):
        if temp is None:
            temp = self.opt.temp
        labels = torch.eye(len(similarity_matrix)).float().cuda()

        # select and combine multiple positives
        pos = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
        # preds = pos
        pos = torch.exp(pos / temp).squeeze(1)

        # select only the negatives the negatives
        neg = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        neg = torch.exp(neg / temp)

        neg_sum = neg.sum(1)

        loss = -(torch.log(pos / (pos + neg_sum)))
        preds = pos / (pos + neg_sum)
        return loss, preds

    def warmup_batch(self, images, captions, lengths, ids, corrs):
        self.step += 1
        batch_length = images.size(0)
        img_embs, cap_embs, cap_lens = self.similarity_model.forward_emb(images, captions, lengths)
        targets_batch = self.similarity_model.targets[ids]
        sims = self.similarity_model.forward_sim(img_embs, cap_embs, cap_lens, 'sim')

        if self.opt.contrastive_loss == 'Triplet':
            loss_cl = self.similarity_model.forward_loss(sims)
        elif self.opt.contrastive_loss == 'InfoNCE':
            loss_cl, preds = self.info_nce_loss(sims)

        loss = loss_cl.mean()

        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss', loss.item(), batch_length)

    def train_batch(self, images, captions, lengths, ids, corrs):
        self.step += 1
        batch_length = images.size(0)

        targets_batch = self.similarity_model.targets[ids]

        img_embs, cap_embs, cap_lens = self.similarity_model.forward_emb(images, captions, lengths)
        sims = self.similarity_model.forward_sim(img_embs, cap_embs, cap_lens, 'sim')
        loss_cl, preds = self.info_nce_loss(sims)
        loss_cl_t, preds_t = self.info_nce_loss(sims.t())
        loss_cl = (loss_cl + loss_cl_t) * 0.5 * targets_batch
        preds = (preds + preds_t) * 0.5

        sims_img, sims_cap = self.similarity_model.sim_enc.forward_topology(img_embs, cap_embs, cap_lens)
        sims_img = torch.sigmoid(sims_img) * targets_batch
        sims_cap = torch.sigmoid(sims_cap) * targets_batch
        sims_topo = sims_img @ sims_cap.t()
        loss_topo, _ = self.info_nce_loss(sims_topo, temp=1.0)
        loss_topo = loss_topo * targets_batch

        self.similarity_model.topos[ids] = torch.cosine_similarity(sims_img, sims_cap, dim=1).data.detach().float()
        self.similarity_model.preds[ids] = preds.data.detach().float()
        target_clean = (targets_batch * corrs).sum() / corrs.sum()
        target_noise = (targets_batch * (1 - corrs)).sum() / (1 - corrs).sum()
        
        loss = loss_cl.mean() + loss_topo.mean() * self.opt.gamma

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
        self.logger.update('Step', self.step)
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])
        self.logger.update('Loss', loss.item(), batch_length)
        self.logger.update('Loss_CL', loss_cl.mean().item(), batch_length)
        self.logger.update('Loss_TOPO', loss_topo.mean().item(), batch_length)
        self.logger.update('Target Clean', target_clean.item(), batch_length)
        self.logger.update('Target Noise', target_noise.item(), batch_length)

    def split_batch(self, corrs=None):
        topos = self.similarity_model.topos
        topos = (topos - topos.min()) / (topos.max() - topos.min())
        
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(topos.detach().cpu().numpy().reshape(-1, 1))
        prob = gmm.predict_proba(topos.detach().cpu().numpy().reshape(-1, 1))
        prob = prob[:, gmm.means_.argmax()]

        self.similarity_model.topo_targets = self.opt.beta * torch.tensor(prob).cuda() + (1 - self.opt.beta) * self.similarity_model.topo_targets
        self.similarity_model.preds_targets = self.opt.beta * self.similarity_model.preds + (1 - self.opt.beta) * self.similarity_model.preds_targets
        self.similarity_model.targets = torch.minimum(self.similarity_model.preds_targets, self.similarity_model.topo_targets)

        if corrs is not None:
            pred = prob > 0.5
            clean_acc = (pred * corrs).sum() / corrs.sum()
            noise_acc = ((1 - pred) * (1 - corrs)).sum() / (1 - corrs).sum()
            print('GMM Clean Acc: ' + str(clean_acc))
            print('GMM Noise Acc: ' + str(noise_acc))