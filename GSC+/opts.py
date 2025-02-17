"""Argument parser"""

import argparse
import os
import random
import sys
import numpy as np
import torch

from utils import save_config


def parse_opt():
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k,cc152k}_precomp')
    parser.add_argument('--data_path', default='/remote-home/share/zhaozh/NC_Datasets/data',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='/remote-home/share/zhaozh/NC_Datasets/vocab',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--folder_name', default='debug', help='Folder name to save the running results')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=60, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=20, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=2e-4, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')

    # ------------------------- model settings (SGR, SAF, and other similarity models) -----------------------#
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SGR', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')

    # ------------------------- our DECL settings -----------------------#
    parser.add_argument('--warmup_if', default=False, type=bool,
                        help='Need warmup training?')
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='Number of warmup epochs.')
    parser.add_argument('--warmup_model_path', default='/remote-home/zhaozh/NC/ELCL_TOPO/runs_1031/f30k_upper/r_corr_4/gamma_001/checkpoint_dir/model_best.pth.tar',
                        help='Path to load a pre-model')
    # noise settings
    parser.add_argument('--noise_ratio', default=0.4, type=float,
                        help='Noisy ratio')
    parser.add_argument('--noise_file', default='/remote-home/share/zhaozh/NC_Datasets/data/noise_file_by_caption/f30k/f30k_precomp_0.4.npy',
                        help='Path to noise index file')
    parser.add_argument("--seed", default=random.randint(0, 100), type=int, help="Random seed.")

    # loss settings
    parser.add_argument('--contrastive_loss', default='InfoNCE', help='Choose in Triplet/InfoNCE/RINCE.')
    parser.add_argument('--temp', default=0.07, type=float, help='Temperature for Info NCE Loss.')
    parser.add_argument('--lam', default=0.01, type=float, help='Parameter for RINCE.')
    parser.add_argument('--q', default=0.1, type=float, help='Parameter for RINCE.')

    # elr settings
    parser.add_argument('--beta', default=0.7, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)

    # gpu settings
    parser.add_argument('--gpu', default='0', help='Which gpu to use.')

    opt = parser.parse_args()
    project_path = str(sys.path[0])
    opt.log_dir = f'{project_path}/runs/{opt.folder_name}/log_dir'
    opt.checkpoint_dir = f'{project_path}/runs/{opt.folder_name}/checkpoint_dir'
    if not os.path.isdir(opt.log_dir):
        os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    # save opt
    save_config(opt, os.path.join(opt.log_dir, "config.json"))

    # set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.random.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = True
    return opt
