import config
import layers

import sys
import torch
import torch.distributions as dist
import random
import utils

from tensorboardX import SummaryWriter
from progress.bar import Bar

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

        self.Encoder = layers.Encoder(num_layers=self.num_layer, 
                                      d_model=self.embedding_dim,
                                      input_vocab_size=self.vocab_size, 
                                      rate=dropout, 
                                      max_len=max_seq
                                      )

        self.Decoder = layers.Decoder(num_layers=self.num_layer, 
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
        src_mask, trg_mask  = utils.get_mask(self.max_seq+1, x, y, config.pad_token)
        memory = self.Encoder(x, mask=src_mask)
        decoder = self.Decoder(y, memory, src_mask, trg_mask)

        fc = self.fc(decoder) # shape: (batch_size, seq_len, vocab_size)
        return fc.contiguous()


    def generate(self,
                 high_input: torch.Tensor,
                 length,
                 tf_board_writer: SummaryWriter = None):
        result_array = torch.tensor([[config.token_sos]])
        memory = self.Encoder(high_input)

        for i in Bar('generating').iter(range(length)):
            if len(result_array) >= config.threshold_len:
                result_array = result_array[:, 1:]

            src_mask, trg_mask = utils.get_mask(len(result_array), high_input, result_array, pad_token=config.pad_token)

            result = self.Decoder(result_array, memory, src_mask, trg_mask)
            result = self.fc(result)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            pdf = dist.OneHotCategorical(probs=result[:, -1])
            result = pdf.sample().argmax(-1).unsqueeze(-1)

            result_array = torch.cat((result_array, result), dim=-1)

        result_array = result_array[0].contiguous().tolist()
        return result_array

