import utils
import random
import pickle
import numpy as np
from splitEncoding import split_encoding, one_hot_sequence
from processor import decode_midi
import config

class DataLoader:
    def __init__(self, high_path, low_path):
        self.X = list(utils.find_files_by_extensions(high_path, ['.pickle']))
        self.Y = list(utils.find_files_by_extensions(low_path, ['.pickle']))
        self.pair = []
        for x, y in zip(self.X, self.Y):
            self.pair.append((x,y))

        # sample = random.sample(self.pair, k = 10)
        # for s in sample:
        #     print(s[0])
        #     print(s[1])
        #     print(" ")
    
        # 0.8 0.1 0.1
        self.pair_dict = {
            'train': self.pair[:int(len(self.pair) * 0.8)],
            'eval': self.pair[int(len(self.pair) * 0.8): int(len(self.pair) * 0.9)],
            'test': self.pair[int(len(self.pair) * 0.9):],
        }

    def batch(self, batch_size, length, mode='train', path=None):
        """
        output: 
            (batch_size, seq_len)
        """
        # random sample and random start place of a sequence
        batch_pair_files = random.sample(self.pair_dict[mode], k=batch_size)
        
        # Max:for debugging, choose the second and third song
        #batch_pair_files = [self.pair_dict[mode][1]]
        batch_x = []
        batch_y = []
        for pair_file in batch_pair_files:
            (x, y) = self._get_seq(pair_file, length, path=path)
            batch_x.append(x)
            batch_y.append(y)
        #batch_x, batch_y = [self._get_seq(pair_file, length) for pair_file in batch_pair_files]

        # print(" ")
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        # print(batch_x.shape)
        # print(type(batch_x))

        return batch_x, batch_y
        
    def _get_seq(self, pair_file, max_length=None, pad_token = config.pad_token, path=None):
        x_file, y_file = pair_file 
        
        if path is not None:
            x_file="./encoded/high/" + path
            y_file="./encoded/low/" + path

        with open(x_file, 'rb') as f:
            #print("File: " + str(x_file))
            x = pickle.load(f)
            #print(len(x))

        with open(y_file, 'rb') as f:
            #print("File: " + str(y_file))
            y = pickle.load(f)

        millisecs = 50000
        #Max: Let's call splitEncoding.
        resX = split_encoding(x, millisecs)
        resY = split_encoding(y, millisecs)

        #print("resX length: " + str(len(resX)))
        #print("resY length: " + str(len(resY)))

        r=np.random.randint(0,min(len(resX), len(resY)))

        x = resX[0]
        y = resY[0]
        # print("First part X: " + str(len(resX[0])))
        # print("First part Y: " + str(len(resY[0])))
    
        if len(x)<2048:
            for i in range(2048-len(x)):
                x.append(pad_token)
        else:
            x = x[0:2048]

        if len(y)<2048:
            for i in range(2048-len(y)):
                y.append(pad_token)
        else:
            y = y[0:2048]  

        decode_midi(x, "./X.midi")
        decode_midi(y, "./Y.midi")

        return (x,y)
        # # cut the data, keep them have the same length (max_length)
        # if max_length is not None:
        #     if max_length <= len(data):
        #         start = random.randrange(0,len(data) - max_length)
        #         data = data[start:start + max_length]
        #     else:
        #         raise IndexError
        # return data



if __name__ == '__main__':

    path = './encoded/'
    dataset = DataLoader( path+'high', path+'low')
    batch_x, batch_y = dataset.batch(1, 2048, path="MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV.midi.pickle")

    print("Size of batch x: " + str(len(batch_x)))
    print("Size of batch y: " + str(len(batch_y)))
    
    print(batch_x)
    # batch_x = np.array(batch_x)
    # batch_y = np.array(batch_y)
    
    """
    batch_x = [one_hot_sequence(seq) for seq in batch_x]
    print(len(batch_x))
    print(batch_x[0].shape)
    """
    """
    Batch_x is a python array of sequences that can be fed into the RNN.
    Each sequence is a 2-D ndarray of dimensions (N_notes, 388)
    Being 388 the size of each one-hot encoded note.
    
    """
    #print(batch_x.shape)
