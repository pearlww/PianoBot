from layers import *
import config

from music_transformer import MusicTransformer
from transformer import Transformer


from data_loader import DataLoader
import utils
from processor import decode_midi, encode_midi
import numpy as np


# def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
#     # remove the start token
#     y = y[1:]
#     bool_acc = 0
#     for i in range(len(y_hat)):
#         if y_hat[i]==y[i]:
#             bool_acc +=1
    
#     return bool_acc/len(y_hat)


# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')


mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=6,
    max_seq=256,
    dropout=0)

  
mt.load_state_dict(torch.load(config.model_dir+'/mtf-seq256.pth'))
mt.eval()

dataset = DataLoader(config.pickle_dir+"high/", config.pickle_dir+"low/")

# test_accuracy =[]
# for i in range(len(dataset.pair_dict['test'])):
#     x, y = dataset.batch(1, config.max_seq, 'test')
#     x = torch.from_numpy(x)
#     y = torch.from_numpy(y).squeeze()
#     y_hat = mt.generate(x)
#     # print(y)
#     # print(y_hat)

#     acc = accuracy(y_hat, y)
#     print(acc)
#     test_accuracy.append(acc)
    

# print("average accuracy:", np.mean(test_accuracy))


inputs = np.array([encode_midi(config.input_midi)[:256]])
print("inputs:", inputs)
targets = np.array([encode_midi(config.target_midi)[:256]])
print("targets:", targets)

results = np.array(mt.generate(torch.from_numpy(inputs)))
print("results:", results)

decode_midi(results, file_path=config.save_path)

