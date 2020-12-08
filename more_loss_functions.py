#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:52:20 2020

@author: max
New loss function
"""

import torch

RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}

#Inputs will have the shape (B x N x V)
#And targets (B X N)
def better_loss(inputs, targets):
    inputs = torch.nn.Softmax(dim=2)(inputs)
    #print(inputs)
    loss = torch.empty(targets.shape)
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            loss[i][j] = negll_classes(inputs[i][j], targets[i][j])
    #print("Loss: " + str(loss))
    return loss


def negll_classes(yhat, y):
    if 0 <= y < START_IDX['note_off']:
        return nsum_nll_range(yhat, 0, START_IDX['note_off'])
    elif START_IDX['note_off'] <= y < START_IDX['time_shift']:
        return nsum_nll_range(yhat, START_IDX['note_off'], START_IDX['time_shift'])
    elif START_IDX['time_shift'] <= y < START_IDX['velocity']:
        return nsum_nll_range(yhat, START_IDX['time_shift'], START_IDX['velocity'])
    elif START_IDX['velocity'] <= y:
        return nsum_nll_range(yhat, START_IDX['velocity'], START_IDX['velocity'] + RANGE_VEL)
    
def nsum_nll_range(yhat, start, end):
    return -torch.sum(torch.log(yhat[start:end])) / (end - start)