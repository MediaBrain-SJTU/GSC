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


def train(opt, data_loader, val_loader, model, epoch, preds=None, mode='warmup', best_rsum=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1
    end = time.time()
    logger.info("=> {mode} epoch: {0}".format(epoch, mode=mode))
    for i, (images, captions, lengths, ids, corrs) in enumerate(data_loader):
        if images.size(0) == 1:
            break
        model.train_start()
        data_time.update(time.time() - end)
        model.logger = train_logger
        ids = torch.tensor(ids).cuda()
        corrs = torch.tensor(corrs).cuda()
        if mode == 'warmup':
            model.warmup_batch(images, captions, lengths, ids, corrs)
        else:
            model.train_batch(images, captions, lengths, ids, corrs)
        batch_time.update(time.time() - end)
        if model.step % opt.log_step == 0:
            logger.info(
                'Epoch ({mode}): [{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t{loss}'.format(epoch, i, num_loader_iter,
                                                                                 mode=mode,
                                                                                 batch_time=batch_time,
                                                                                 data_time=data_time,
                                                                                 loss=str(model.logger)))
        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.step)
        tb_logger.log_value('step', i, step=model.step)
        tb_logger.log_value('batch_time', batch_time.val, step=model.step)
        tb_logger.log_value('data_time', data_time.val, step=model.step)
        model.logger.tb_log(tb_logger, step=model.step)
    
    model.split_batch(train_loader.dataset.corrs)


def validation(opt, val_loader, model, test=False):
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
    img_embs, cap_embs, cap_lens = encode_data(model.similarity_model, val_loader, opt.log_step)

    # clear duplicate 5*images and keep 1*images
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])

    # record computation time of validation
    start = time.time()
    sims = shard_attn_scores(model.similarity_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000)
    end = time.time()
    logger.info(f"calculate similarity time: {end - start}")

    with torch.no_grad():
        loss_cl, preds = model.info_nce_loss(torch.tensor(sims[:, :1000]).float().cuda())
        loss_cl_t, preds_t = model.info_nce_loss(torch.tensor(sims[:, :1000]).float().cuda().t())
        loss_cl = (loss_cl + loss_cl_t) * 0.5

        sims_img, sims_cap = model.similarity_model.sim_enc.forward_topology(torch.tensor(img_embs[:128]).float().cuda(), torch.tensor(cap_embs[:128]).float().cuda(), cap_lens[:128])
        sims_img = torch.sigmoid(sims_img)
        sims_cap = torch.sigmoid(sims_cap)
        sims_topo = sims_img @ sims_cap.t()
        loss_topo, _ = model.info_nce_loss(sims_topo, temp=1.0)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims, per_captions, return_ranks=False)
    logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
    logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims, per_captions, return_ranks=False)
    logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
    logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    logger.info("Sum of Recall: %.2f" % (r_sum))
    # sum of recalls to be used for early stopping
    if test:
        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.step)
        tb_logger.log_value('r5', r5, step=model.step)
        tb_logger.log_value('r10', r10, step=model.step)
        tb_logger.log_value('medr', medr, step=model.step)
        tb_logger.log_value('meanr', meanr, step=model.step)
        tb_logger.log_value('r1i', r1i, step=model.step)
        tb_logger.log_value('r5i', r5i, step=model.step)
        tb_logger.log_value('r10i', r10i, step=model.step)
        tb_logger.log_value('medri', medri, step=model.step)
        tb_logger.log_value('meanr', meanr, step=model.step)
        tb_logger.log_value('r_sum', r_sum, step=model.step)
        tb_logger.log_value('loss_cl', loss_cl.mean().item(), step=model.step)
        tb_logger.log_value('loss_topo', loss_topo.mean().item(), step=model.step)
    else:
        # record metrics in tensorboard
        tb_logger.log_value('t-r1', r1, step=model.step)
        tb_logger.log_value('t-r5', r5, step=model.step)
        tb_logger.log_value('t-r10', r10, step=model.step)
        tb_logger.log_value('t-medr', medr, step=model.step)
        tb_logger.log_value('t-meanr', meanr, step=model.step)
        tb_logger.log_value('t-r1i', r1i, step=model.step)
        tb_logger.log_value('t-r5i', r5i, step=model.step)
        tb_logger.log_value('t-r10i', r10i, step=model.step)
        tb_logger.log_value('t-medri', medri, step=model.step)
        tb_logger.log_value('t-meanr', meanr, step=model.step)
        tb_logger.log_value('t-r_sum', r_sum, step=model.step)
        tb_logger.log_value('t-loss_cl', loss_cl.mean().item(), step=model.step)
        tb_logger.log_value('t-loss_topo', loss_topo.mean().item(), step=model.step)
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
    model = RINCE(opt)
    best_rsum = 0
    start_epoch = 0
    if opt.warmup_if:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model.load_state_dict(checkpoint['model'])
            logger.info(
                "=> Load warmup(pre-) checkpoint '{}' (epoch {})".format(opt.warmup_model_path, checkpoint['epoch']))
            if 'best_rsum' in checkpoint:
                if 'warmup' not in opt.warmup_model_path:
                    start_epoch = checkpoint['epoch'] + 1
                best_rsum = checkpoint['best_rsum']
            model.step = checkpoint['step']
        else:
            logger.info(f"=> no checkpoint found at '{opt.warmup_model_path}', warmup start!")
            for e in range(opt.warmup_epochs):
                train(opt, train_loader, val_loader, model, e, mode='warmup')
                save_checkpoint({
                    'epoch': e,
                    'model': model.state_dict(),
                    'opt': opt,
                    'step': model.step
                }, is_best=False, filename='warmup_model_{}.pth.tar'.format(e), prefix=opt.checkpoint_dir + '/')
    else:
        logger.info("=> No warmup stage")

    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)
        start_time = datetime.now()
        train(opt, train_loader, val_loader, model, epoch, mode='train', best_rsum=best_rsum)
        end_time = datetime.now()
        tb_logger.log_value('cost_time', int((end_time - start_time).seconds), step=epoch)
        validation(opt, test_loader, model, test=True)
        rsum = validation(opt, val_loader, model)
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'step': model.step,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.checkpoint_dir + '/')
