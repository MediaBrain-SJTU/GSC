import json
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='',ckpt=True):
    tries = 15
    error = None
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            if ckpt:
                torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)


def load_config(opt, file_path):
    with open(file_path, "r") as f:
        opt.__dict__ = json.load(f)
