import utils
import random
import pickle
import numpy as np
from splitEncoding import split_encoding, crop_pad_sequences, get_sequences, pad_sequence
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

    def batch(self, batch_size, length, mode='train', path=None, min_seq=config.min_seq):
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
            try:
                (x, y) = self._get_seq(pair_file, length, path=path, min_seq=min_seq)
            except Exception:
                #If it failed, try another file
                return self.batch(batch_size, length, mode, path, min_seq)
            batch_x.append(x)
            batch_y.append(y)
        #batch_x, batch_y = [self._get_seq(pair_file, length) for pair_file in batch_pair_files]

        # print(" ")
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        # print(batch_x.shape)
        # print(type(batch_x))

        return batch_x, batch_y
        
    #Gets an X and Y sequence of required length from pair_file (or alternatively, from path if it's not none)
    def _get_seq(self, pair_file, max_length, pad_token = config.pad_token, path=None, min_seq=config.min_seq):
        x_file, y_file = pair_file 
        
        if path is not None:
            x_file="./encoded/high/" + path
            y_file="./encoded/low/" + path

        with open(x_file, 'rb') as f:
            #print("File: " + str(x_file))
            x = pickle.load(f)

        with open(y_file, 'rb') as f:
            #print("File: " + str(y_file))
            y = pickle.load(f)

        #SANITY CHECK: The file has enough data
        if len(x) < min_seq:
            print("FILE X TOO SHORT:")
            print(x_file)
            raise Exception
        if len(y) < min_seq:
            print("FILE Y TOO SHORT:")
            print(y_file)
            raise Exception
        
        """
        DEPRECATED: Cut by time. Now we're only gonna cut by sequence length
        millisecs = 50000
        #Max: Let's call splitEncoding.
        resX = split_encoding(x, millisecs)
        resY = split_encoding(y, millisecs)

        # print("resX length: " + str(len(resX)))
        # print("resY length: " + str(len(resY)))
        
        
        r=np.random.randint(0,min(len(resX), len(resY)))
        
        x = resX[r]
        y = resY[r]
        # print("First part X: " + str(len(resX[0])))
        # print("First part Y: " + str(len(resY[0])))
        
        
        if len(x)<2048:
            for i in range(2048-len(x)):
                x.append(pad_token)
        elif len(x)>2048:
            x = x[0:2048]

        if len(y)<2048:
            for i in range(2048-len(y)):
                y.append(pad_token)
        else:
            y = y[0:2048]  
        
        x, y = crop_pad_sequences(x, y, pad_token, max_length)
        
        """
        
        #How many sequences can we get?
        nsequences = int((len(x) // max_length) + 1)
        #I want to select one of them. I make a random ordering.
        #If the first one fails, I'll take the next one and so on.
        positions = random.sample(list(range(0,nsequences)), nsequences) #Basically a random permutation
        
        seqX, seqY = get_sequences(x, y, max_length, positions[0])
        #If it failed, try again with another sequence
        i = 1
        while (i < nsequences) and ((len(seqX) < min_seq) or (len(seqY) < min_seq)):
            seqX, seqY = get_sequences(x, y, max_length, positions[i])
            i+=1
        #print("Position: ", positions[i-1])
        if i==nsequences:
            #None of the sequences work! Fail!
            print("Failed to get any min_length sequence!!!")
            print(x_file)
            print(y_file)
            raise Exception
            
        """
        
        decode_midi(x, "./X.midi")
        decode_midi(y, "./Y.midi")
        
        """
        
        #Else we've got the sequences, but without padding
        #PAD
        seqX = pad_sequence(seqX, pad_token, max_length, add_eos=False)
        seqY = pad_sequence(seqY, pad_token, max_length, add_eos=True)
        seqY.insert(0, config.token_sos)
        
        return (seqX, seqY)
    
    def check(self):
        filter_bad(self.pair_dict['train'])
        filter_bad(self.pair_dict['eval'])
        filter_bad(self.pair_dict['test'])

def filter_bad(files):
    for x_file, y_file in files:
        with open(x_file, 'rb') as f:
            #print("File: " + str(x_file))
            x = pickle.load(f)
            if len(x) < config.min_seq:
                print("TOO SHORT: ", x_file)
                

        with open(y_file, 'rb') as f:
                #print("File: " + str(y_file))
            y = pickle.load(f)
            if len(y) < config.min_seq:
                print("TOO SHORT: ", y_file)
            

def test():
    file="/home/max/Documents/DTU/Deep Learning/Project/MusicTransformer-pytorch/originalEncoding/MIDI-Unprocessed_01_R1_2009_01-04_ORIG_MID--AUDIO_01_R1_2009_01_R1_2009_01_WAV.midi.pickle"
    with open(file, 'rb') as f:
        x = pickle.load(f)
        decode_midi(x, "./test/originalDecoded.midi")
    
    

if __name__ == '__main__':

    path = './encoded/'
    dataset = DataLoader( path+'high', path+'low')
    batch_x, batch_y = dataset.batch(4, 128)#, path="MIDI-Unprocessed_17_R2_2011_MID--AUDIO_R2-D5_03_Track03_wav.midi.pickle")
    
    print("Size of batch x: " + str(len(batch_x)))
    print("Size of batch y: " + str(len(batch_y)))
    print("Batch x:")
    print(batch_x)
    print("Batch y:")
    print(batch_y)
    #batch_x = np.array(batch_x)
    #batch_y = np.array(batch_y)
    
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
