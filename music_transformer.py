import config
import layers

import sys
import torch
import torch.distributions as dist
import random
import utils

from tensorboardX import SummaryWriter
from progress.bar import Bar

from splitEncoding import pad_sequence

class MusicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size, num_layer,
                 max_seq, dropout, dist=False, writer=None):
        super().__init__()

        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.dist = dist
        self.writer = writer

        self.Encoder = layers.EncoderMusic(num_layers=self.num_layer, 
                                      d_model=self.embedding_dim,
                                      input_vocab_size=self.vocab_size, 
                                      rate=dropout, 
                                      max_len=max_seq
                                      )

        self.Decoder = layers.DecoderMusic(num_layers=self.num_layer, 
                                      d_model=self.embedding_dim,
                                      input_vocab_size=self.vocab_size, 
                                      rate=dropout, 
                                      max_len=max_seq+1
                                      )

        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x, y, length=None, writer=None):
        """
        Args:
            x: (batch_size, seq_len)
            y: (batch_size, seq_len+1)
        Returns:
            output: (batch_size, seq_len, vocab_size)
        """
        src_mask, trg_mask  = utils.get_4d_mask(self.max_seq+1, x, y, config.pad_token)
        # src_mask = utils.get_src_mask(x, config.pad_token)
        # trg_mask = utils.get_tgt_mask(y,config.pad_token)

        memory = self.Encoder(x, mask=src_mask)
        decoder = self.Decoder(y, memory, src_mask, trg_mask)

        fc = self.fc(decoder) # shape: (batch_size, seq_len, vocab_size)
        return fc.contiguous()


    def generate(self,
                 input: torch.Tensor,
                 max_length = config.max_length,
                 tf_board_writer: SummaryWriter = None):

        result_array = torch.tensor([[config.token_sos]])
        # #Max: I don't vibe with variable lengths. But I do vibe with padding :)
        # result_array = pad_sequence([config.token_sos], config.pad_token, maxint=self.max_seq+1)
        # result_array = torch.tensor([result_array])
        #print(result_array)

        #I really think here we should put the source mask. Why not?
        #Let's try
        src_mask = utils.get_src_mask(input, config.pad_token)
        memory = self.Encoder(input, mask=src_mask)

        for i in Bar('generating').iter(range(max_length)):

            #Why len(result_array) and not self.max_seq+1? Because it makes sense.
            # src_mask, trg_mask = utils.get_4d_mask(i, input, result_array, pad_token=config.pad_token)
            trg_mask = utils.get_tgt_mask(result_array,config.pad_token)

            result = self.Decoder(result_array, memory, src_mask, trg_mask)
            result = self.fc(result)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            pdf = dist.OneHotCategorical(probs=result[:, -1])
            result = pdf.sample().argmax(-1).unsqueeze(-1)

            if result == config.token_eos:
                break
            result_array = torch.cat((result_array, result), dim=-1)

            # #Instead of appending, substitute a padding
            # if (i+1)==len(result_array[-1]):
            #     result_array = torch.cat((result_array, result), dim=-1)
            # else:
            #     result_array[-1][i+1]=result

        result_array = result_array[0]
        return result_array[1:]

