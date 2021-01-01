import torch

# data
pickle_dir = './encoded/'


# train
device = torch.device('cpu')

epochs = 100 # 100 for debug
batch_size = 1 # 2 for debag
dropout = 0.1
debug = 'true'
l_r = 0.0001
label_smooth = 0.1


# model
experiment = 'embedding256-layer2'
max_seq = 128
embedding_dim = 256 #512
num_layers = 2
event_dim = 388


pad_token = event_dim 
token_sos = event_dim + 1
token_eos = event_dim + 2

vocab_size = event_dim + 3

model_dir = "./train"


# generate
input_midi = './output/high/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi'
target_midi = './output/low/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_06_Track06_wav.midi'
save_path = 'bin/generated.mid'
length =  128
threshold_len = 128
