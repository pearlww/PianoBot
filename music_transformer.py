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

        self.infer = False

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
        if not self.infer:
            src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq+1, x, y, config.pad_token)
            memory = self.Encoder(x, mask=src_mask)
            decoder = self.Decoder(y, memory, mask=look_ahead_mask)#|trg_mask) # shape: (batch_size, seq_len, embedding_dim)

            fc = self.fc(decoder) # shape: (batch_size, seq_len, vocab_size)
            return fc.contiguous()
        else:
            return self.generate(x, length, None).contiguous().tolist()

    def generate(self,
                 high_input: torch.Tensor,
                 length,
                 tf_board_writer: SummaryWriter = None):
        decode_array = high_input
        result_array = torch.tensor([])

        for i in Bar('generating').iter(range(length)):
            if decode_array.size(1) >= config.threshold_len:
                decode_array = decode_array[:, 1:]
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.size(1), decode_array, decode_array, pad_token=config.pad_token)

            result, _ = self.Decoder(decode_array, None)
            result = self.fc(result)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)


            pdf = dist.OneHotCategorical(probs=result[:, -1])
            result = pdf.sample().argmax(-1).unsqueeze(-1)
            decode_array = torch.cat((decode_array, result), dim=-1)
            result_array = torch.cat((result_array, result), dim=-1)

        result_array = result_array[0]
        return result_array

    def test(self):
        self.eval()
        self.infer = True
