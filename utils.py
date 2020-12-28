import os
import numpy as np
#from deprecated.sequence import EventSeq, ControlSeq
import torch
import torch.nn.functional as F
import torchvision

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)


def sequence_mask(length, max_length=None):

    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_mask(size, src, trg, pad_token):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :param pad_token: pad token

    :return: src_mask hides padding
             target_mask hides padding and future words
    """

    src = src[:, None, None, :] # (batch_size, 1, 1, seq_length)
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = src == src_pad_tensor

    trg = trg[:, None, None, :]
    trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
    trg_mask = trg == trg_pad_tensor

    # boolean reversing i.e) True * -1 + 1 = False
    seq_mask = ~sequence_mask(torch.arange(1, size+1).to(trg.device), size)
    # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
    trg_mask  = trg_mask | seq_mask

    return src_mask, trg_mask

def get_src_mask(size, src, pad_token):
    #Max: First three lines of get_mask    
    src = src[:, None, None, :] # (batch_size, 1, 1, seq_length)
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = src == src_pad_tensor
    return src_mask

if __name__ == '__main__':

    s = np.array([np.array([1, 2]*50),np.array([1, 2, 3, 4]*25)])

    t = np.array([np.array([2, 3, 4, 5, 6]*20), np.array([1, 2, 3, 4, 5]*20)])
    print(t.shape)

    print(get_mask(100, s, t, pad))