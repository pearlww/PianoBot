from layers import *
import config

from music_transformer import MusicTransformer
from data_loader import DataLoader
import utils
from processor import decode_midi, encode_midi

import datetime
from tensorboardX import SummaryWriter


# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')

# current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
# gen_summary_writer = SummaryWriter(gen_log_dir)


mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0)

  
mt.load_state_dict(torch.load(config.model_dir+'/music-final.pth'))
mt.eval()

inputs = np.array([encode_midi(config.input_midi)[:128]])
print("inputs:", inputs)
targets = np.array([encode_midi(config.target_midi)[:128]])
print("targets:", targets)

results = np.array(mt.generate(torch.from_numpy(inputs)))
print("results:", results)


# def remove_padding(results):
#     ret = []
#     for c in results:
#         if c!=388:
#             ret.append(c)
#     return ret

# results = remove_padding(results)
# print("outputs without pad:", result)

decode_midi(results, file_path=config.save_path)

# gen_summary_writer.close()
