#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:58:17 2020

@author: max

Trying to split a dataset using the mido library
"""

from mido import MidiFile, Message
import os    #To traverse the file tree


"""
Loads the midi file in path with mido, replace the upper
(lower) notes with rests and save the ouptut in the output directory
"""


full=True

def split(inputPath, outputPath, keep_up=True):
    mid = MidiFile(inputPath)
    for i, track in enumerate(mid.tracks):
        for i, msg in enumerate(track):
            if msg.type == 'note_on' and msg.note>=64:
                rest=Message('note_off')
                rest.velocity=msg.velocity
                rest.time=msg.time
                track[i]=rest
    
    mid.save(outputPath)
            
    

parentDirectory = "/home/max/Documents/DTU/Deep Learning/Project/maestro-v2.0.0/"
outputDirectory = "/home/max/Documents/DTU/Deep Learning/Project/output/outputLow/"
outputDirectories=[]
paths={}

if full:
    for root, dirs, files in os.walk(parentDirectory):
        if root == parentDirectory:
            for d in dirs:
                if not os.path.exists(outputDirectory+"/"+str(d)):
                    os.makedirs(outputDirectory+"/"+str(d))
        else:
            for f in files:    
            #We're in one sub-folder, where the files are
                if not (root in paths):
                    paths[root] = []
                paths[root].append(f)

    for d, paths in paths.items():
        for path in paths:
            split(d+"/"+path, outputDirectory+"/"+d[-4:]+"/"+path)
else:
    split("/home/max/Documents/DTU/Deep Learning/Project/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi",
          "/home/max/Documents/DTU/Deep Learning/Project/output/outputHigh/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")