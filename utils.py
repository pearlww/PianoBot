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
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def get_masked_with_pad_tensor(size, src, trg, pad_token):
    """
    :param size: the size of target input
    :param src: source tensor
    :param trg: target tensor
    :param pad_token: pad token
    :return:
    """
    src = src[:, None, None, :]
    trg = trg[:, None, None, :]
    src_pad_tensor = torch.ones_like(src).to(src.device.type) * pad_token
    src_mask = torch.equal(src, src_pad_tensor)
    trg_mask = torch.equal(src, src_pad_tensor)
    if trg is not None:
        trg_pad_tensor = torch.ones_like(trg).to(trg.device.type) * pad_token
        dec_trg_mask = trg == trg_pad_tensor
        # boolean reversing i.e) True * -1 + 1 = False
        seq_mask = ~sequence_mask(torch.arange(1, size+1).to(trg.device), size)
        # look_ahead_mask = torch.max(dec_trg_mask, seq_mask)
        look_ahead_mask = dec_trg_mask | seq_mask

    else:
        trg_mask = None
        look_ahead_mask = None

    return src_mask, trg_mask, look_ahead_mask