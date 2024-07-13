"""Training script"""

import logging
import os
import time
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import tensorboard_logger as tb_logger
import data
import opts
from model.RINCE import RINCE
from evaluation import encode_data, shard_attn_scores, i2t, t2i, AverageMeter, LogCollector
from utils import save_checkpoint
from vocab import deserialize_vocab
import warnings

warnings.filterwarnings("ignore")

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.2 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(opt, data_loader, val_loader, model_A, model_B, epoch, preds=None, mode='warmup', best_rsum=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1
    end = time.time()
    logger.info("=> {mode} epoch: {0}".format(epoch, mode=mode))
    for i, (images, captions, lengths, ids, corrs) in enumerate(data_loader):
        # print(i)
        if images.size(0) == 1:
            break
        model_A.train_start()
        model_B.train_start()
        data_time.update(time.time() - end)
        model_A.logger = train_logger
        model_B.logger = train_logger
        ids = torch.tensor(ids).cuda()
        corrs = torch.tensor(corrs).cuda()
        if mode == 'warmup':
            model_A.warmup_batch(images, captions, lengths)
            model_B.warmup_batch(images, captions, lengths)
        else:
            model_A.train_batch(images, captions, lengths, ids, corrs, model_B.similarity_model.targets[ids])
            model_B.train_batch(images, captions, lengths, ids, corrs, model_A.similarity_model.targets[ids])
        batch_time.update(time.time() - end)
        if model_A.step % opt.log_step == 0:
            logger.info(
                'Epoch ({mode}): [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t{loss}'.format(epoch, i, num_loader_iter,
                                                                                 mode=mode,
                                                                                 batch_time=batch_time,
                                                                                 data_time=data_time,
                                                                                 loss=str(model_A.logger)))
        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model_A.step)
        tb_logger.log_value('step', i, step=model_A.step)
        tb_logger.log_value('batch_time', batch_time.val, step=model_A.step)
        tb_logger.log_value('data_time', data_time.val, step=model_A.step)
        model_A.logger.tb_log(tb_logger, step=model_A.step)
        model_B.logger.tb_log(tb_logger, step=model_B.step)
    
    if mode != 'warmup':
        model_A.split_batch(train_loader.dataset.corrs)
        model_B.split_batch(train_loader.dataset.corrs)


def validation(opt, val_loader, model_A, model_B, test=False):
    # compute the encoding for all the validation images and captions
    if opt.data_name == 'cc152k_precomp':
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        logger.info(f"No dataset")
        return 0
    if test:
        logger.info(f"=> Test")
    else:
        logger.info(f"=> Validation")
    # compute the encoding for all the validation images and captions
    img_embs_A, cap_embs_A, cap_lens_A = encode_data(model_A.similarity_model, val_loader, opt.log_step)
    img_embs_A = np.array([img_embs_A[i] for i in range(0, len(img_embs_A), per_captions)])

    img_embs_B, cap_embs_B, cap_lens_B = encode_data(model_B.similarity_model, val_loader, opt.log_step)
    img_embs_B = np.array([img_embs_B[i] for i in range(0, len(img_embs_B), per_captions)])


    # record computation time of validation
    start = time.time()
    sims_A = shard_attn_scores(model_A.similarity_model, img_embs_A, cap_embs_A, cap_lens_A, opt, shard_size=1000)
    end = time.time()
    logger.info(f"calculate similarity time: {end - start}")

    start = time.time()
    sims_B = shard_attn_scores(model_B.similarity_model, img_embs_B, cap_embs_B, cap_lens_B, opt, shard_size=1000)
    end = time.time()
    logger.info(f"calculate similarity time: {end - start}")
    sims = (sims_A + sims_B) / 2

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs_A.shape[0], sims, per_captions, return_ranks=False)
    logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
    logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs_A.shape[0], sims, per_captions, return_ranks=False)
    logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
    logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    logger.info("Sum of Recall: %.2f" % (r_sum))
    # sum of recalls to be used for early stopping
    if test:
        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model_A.step)
        tb_logger.log_value('r5', r5, step=model_A.step)
        tb_logger.log_value('r10', r10, step=model_A.step)
        tb_logger.log_value('medr', medr, step=model_A.step)
        tb_logger.log_value('meanr', meanr, step=model_A.step)
        tb_logger.log_value('r1i', r1i, step=model_A.step)
        tb_logger.log_value('r5i', r5i, step=model_A.step)
        tb_logger.log_value('r10i', r10i, step=model_A.step)
        tb_logger.log_value('medri', medri, step=model_A.step)
        tb_logger.log_value('meanr', meanr, step=model_A.step)
        tb_logger.log_value('r_sum', r_sum, step=model_A.step)
    else:
        # record metrics in tensorboard
        tb_logger.log_value('t-r1', r1, step=model_A.step)
        tb_logger.log_value('t-r5', r5, step=model_A.step)
        tb_logger.log_value('t-r10', r10, step=model_A.step)
        tb_logger.log_value('t-medr', medr, step=model_A.step)
        tb_logger.log_value('t-meanr', meanr, step=model_A.step)
        tb_logger.log_value('t-r1i', r1i, step=model_A.step)
        tb_logger.log_value('t-r5i', r5i, step=model_A.step)
        tb_logger.log_value('t-r10i', r10i, step=model_A.step)
        tb_logger.log_value('t-medri', medri, step=model_A.step)
        tb_logger.log_value('t-meanr', meanr, step=model_A.step)
        tb_logger.log_value('t-r_sum', r_sum, step=model_A.step)
    return r_sum

def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    tb_logger.configure(opt.log_dir, flush_secs=5)
    logger = init_logging(opt.log_dir + '/log.txt')
    logger.info(opt)
    logger.info(f"=> PID:{os.getpid()}, GPU:[{opt.gpu}], Noise ratio: {opt.noise_ratio}")
    logger.info(f"=> Log save path: '{opt.log_dir}'")
    logger.info(f"=> Checkpoint save path: '{opt.checkpoint_dir}'")
    # Load Vocabulary
    logger.info(f"=> Load vocabulary from '{opt.vocab_path}'")
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)
    # Load data loaders
    logger.info(f"=> Load loaders from '{opt.data_path}/{opt.data_name}'")
    train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size,
                                                             opt.workers, opt)
    # Construct the model (DECL-)
    logger.info(f"=> Similarity model is {opt.module_name}")
    model_A = RINCE(opt)
    model_B = RINCE(opt)
    best_rsum = 0
    start_epoch = 0
    if opt.warmup_if:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model_A.load_state_dict(checkpoint['model_A'])
            model_B.load_state_dict(checkpoint['model_B'])
            logger.info(
                "=> Load warmup(pre-) checkpoint '{}' (epoch {})".format(opt.warmup_model_path, checkpoint['epoch']))
            if 'best_rsum' in checkpoint:
                if 'warmup' not in opt.warmup_model_path:
                    start_epoch = checkpoint['epoch'] + 1
                best_rsum = checkpoint['best_rsum']
            model_A.step = checkpoint['step']
            model_B.step = checkpoint['step']
        else:
            logger.info(f"=> no checkpoint found at '{opt.warmup_model_path}', warmup start!")
            for e in range(opt.warmup_epochs):
                train(opt, train_loader, val_loader, model_A, model_B, e, mode='warmup')
                save_checkpoint({
                    'epoch': e,
                    'model_A': model_A.state_dict(),
                    'model_B': model_B.state_dict(),
                    'opt': opt,
                    'step': model_A.step
                }, is_best=False, filename='warmup_model_{}.pth.tar'.format(e), prefix=opt.checkpoint_dir + '/')
    else:
        logger.info("=> No warmup stage")

    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)
        start_time = datetime.now()
        train(opt, train_loader, val_loader, model_A, model_B, epoch, mode='train', best_rsum=best_rsum)
        end_time = datetime.now()
        tb_logger.log_value('cost_time', int((end_time - start_time).seconds), step=epoch)
        validation(opt, test_loader, model_A, model_B, test=True)
        rsum = validation(opt, val_loader, model_A, model_B)
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch,
            'model_A': model_A.state_dict(),
            'model_B': model_B.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'step': model_A.step,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.checkpoint_dir + '/')
