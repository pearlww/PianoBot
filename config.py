import torch

pickle_dir = './encoded/'

device = torch.device('cpu')


# train
epochs = 100 # 100 for debug
batch_size = 2 # 2 for debag
dropout = 0.1
debug = 'true'
l_r = 0.001
label_smooth = 0.1



# model
experiment = 'embedding256-layer6'
max_seq = 2048
embedding_dim = 256
num_layers = 2
event_dim = 388

pad_token = event_dim
token_sos = event_dim + 1
token_eos = event_dim + 2
vocab_size = event_dim + 3

model_dir = "."

# experiment: 'embedding512-layer6'
# max_seq: 2048
# embedding_dim: 512
# num_layers: 6
# event_dim: 388 


# generate
save_path: 'bin/generated.mid'
length: 4000
threshold_len: 500
