import utils
import random
import pickle
import numpy as np

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

    def batch(self, batch_size, length, mode='train'):
        """
        output: 
            (batch_size, seq_len)
        """
        # random sample and random start place of a sequence
        batch_pair_files = random.sample(self.pair_dict[mode], k=batch_size)
        batch_x, batch_y = [self._get_seq(pair_file, length) for pair_file in batch_pair_files]

        return batch_x, batch_y 
        
    def _get_seq(self, pair_file, max_length=None):
        x_file, y_file = pair_file 

        with open(x_file, 'rb') as f:
            x = pickle.load(f)

        with open(y_file, 'rb') as f:
            y = pickle.load(f)

        print(len(x))
        print(len(y))
        # # cut the data, keep them have the same length (max_length)
        # if max_length is not None:
        #     if max_length <= len(data):
        #         start = random.randrange(0,len(data) - max_length)
        #         data = data[start:start + max_length]
        #     else:
        #         raise IndexError
        # return data




if __name__ == '__main__':

    path = '/home/pearl/myGit/PianoBot/data_encoded/'
    dataset = DataLoader( path+'high', path+'low')
    batch_x, batch_y = dataset.batch(2, 2048)
    #print(batch_x.shape)