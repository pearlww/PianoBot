import torch

# data
pickle_dir = './encoded/'


# train
device = torch.device('cpu')

epochs = 100 # 100 for debug
batch_size = 16 # 2 for debag
dropout = 0.1
debug = 'true'
l_r = 0.0001
label_smooth = 0.1


# model
experiment = 'tf-seq128-layer3'

max_seq = 128
min_seq = 16
embedding_dim = 256 #512
num_layers = 6
event_dim = 388



pad_token = event_dim 
token_sos = event_dim + 1
token_eos = event_dim + 2

vocab_size = event_dim + 3

model_dir = "./models"


# generate
input_midi = './split/high/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--3.midi'
target_midi = './split/low/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--3.midi'
save_path = './output/generated.mid'
max_length =  256
