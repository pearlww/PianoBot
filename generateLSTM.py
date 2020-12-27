from layers import *
import config

from music_transformer import MusicTransformer
from data_loader import DataLoader
import utils
from processor import decode_midi, encode_midi

import datetime
from tensorboardX import SummaryWriter
from splitEncoding import one_hot_sequence
from LSTM import MusicLSTM

# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)


mt = MusicLSTM()

  
mt.load_state_dict(torch.load(config.model_dir+'/final.pth'))
mt.eval()

inputs = np.array([one_hot_sequence(encode_midi(config.input_midi)[:128])])
#print("inputs:", inputs)
targets = np.array([encode_midi(config.target_midi)[:128]])
#print("targets:", targets)


inputs = torch.from_numpy(inputs)
result = mt.generate(inputs, config.length, gen_summary_writer)
print("outputs", result)

def remove_padding(result):
    ret = []
    for c in result:
        if c!=388:
            ret.append(c)
    return ret

result = remove_padding(result)
print("outputs without pad:", result)

# decode_midi(result, file_path=config.save_path)

# gen_summary_writer.close()
