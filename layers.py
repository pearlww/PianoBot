import utils

import math as m
import numpy as np
import math, copy
import config
import torch
import torch.nn.functional as F


def sinusoid(max_seq, embedding_dim):
    return np.array([[
        [
            m.sin(
                pos * m.exp(-m.log(10000) * i / embedding_dim) * m.exp(
                    m.log(10000) / embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        # print(x.shape)
        # print(x.size(1))
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = self.clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        
    def forward(self, inputs, mask=None):
        "Implements Figure 2"
        query, key, value = inputs

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)

        return torch.matmul(p_attn, value), p_attn
    def clones(self, module, N):
        "Produce N identical layers."
        return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)

        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        #print(mask)
        if mask is not None:
            logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = utils.sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.sa = MultiHeadedAttention(h=h, d_model=d_model)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model//2)
        self.FFN_suf = torch.nn.Linear(self.d_model//2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        attn_out = self.sa([x,x,x], mask)

        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model

        self.sa2 = MultiHeadedAttention(h=h, d_model=d_model)
        self.sa = MultiHeadedAttention(h=h, d_model=d_model)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = torch.nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, src_mask=None, tgt_mask=None, **kwargs):
        #print("Forwarding a decoder layer")

        attn_out = self.sa([x,x,x], mask=tgt_mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        attn_out2 = self.sa2([out1, encode_out, encode_out], mask=src_mask)
        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1+attn_out2)

        ffn_out = F.relu(self.FFN_pre(attn_out2))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2+ffn_out)

        return out

class EncoderMusicLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderMusicLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model//2)
        self.FFN_suf = torch.nn.Linear(self.d_model//2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        attn_out, w = self.rga([x,x,x], mask)


        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2


class DecoderMusicLayer(torch.nn.Module):
    def __init__(self, d_model, rate=0.1, h=16, additional=False, max_seq=2048):
        super(DecoderMusicLayer, self).__init__()

        self.d_model = d_model
        self.rga2 = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)
        self.rga = RelativeGlobalAttention(d=d_model, h=h, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, self.d_model // 2)
        self.FFN_suf = torch.nn.Linear(self.d_model // 2, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, src_mask=None, tgt_mask=None, **kwargs):
        #print("Forwarding a decoder layer")

        attn_out, aw1 = self.rga([x, x, x], mask=tgt_mask)

        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)


        attn_out2, aw2 = self.rga2([out1, encode_out, encode_out], mask=src_mask)

        attn_out2 = self.dropout2(attn_out2)
        attn_out2 = self.layernorm2(out1+attn_out2)

        ffn_out = F.relu(self.FFN_pre(attn_out2))
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        out = self.layernorm3(attn_out2+ffn_out)

        return out

class Encoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model, padding_idx=config.pad_token)
        self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        """
        input x:  (batch_size, seq_len)

        output: (batch_size, seq_len, embedding_dim)
        """
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model, padding_idx=config.pad_token)
        self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, src_mask=None, tgt_mask=None):

        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, encode_out, src_mask, tgt_mask)
        return x


class EncoderMusic(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(EncoderMusic, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model, padding_idx=config.pad_token)
        self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = torch.nn.ModuleList(
            [EncoderMusicLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask=None):
        """
        input x:  (batch_size, seq_len)

        output: (batch_size, seq_len, embedding_dim)
        """
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        return x


class DecoderMusic(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(DecoderMusic, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model, padding_idx=config.pad_token)
        self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.dec_layers = torch.nn.ModuleList(
            [DecoderMusicLayer(d_model, rate, h=self.d_model // 64, additional=False, max_seq=max_len)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, src_mask=None, tgt_mask=None):

        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, encode_out, src_mask, tgt_mask)
        return x


class BasicEncoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(BasicEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = torch.nn.ModuleList(
            [torch.nn.TransformerEncoderLayer(d_model=d_model,nhead=8,dim_feedforward=max_len,dropout=rate)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, src_mask=None, src_key_mask=None):
        """
        input x:  (batch_size, seq_len)

        output: (batch_size, seq_len, embedding_dim)
        """
        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        #RESHAPE FOR PYTORCH: BxLxV -> LxBxV
        x = x.permute(1,0,2)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, src_mask=src_mask, src_key_padding_mask=src_key_mask)
            if torch.isnan(x[0][0][0]):
                print("Encoder nan")
            
        #RESHAPE FOR OUTPUT: LxBxV -> BxLxV
        x = x.permute(1,0,2)
        return x


class BasicDecoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(BasicDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=d_model)
        if True:
            self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.dec_layers = torch.nn.ModuleList(
            [torch.nn.TransformerDecoderLayer(d_model=d_model,nhead=8,dim_feedforward=max_len,dropout=rate)
             for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, encode_out, memory_mask=None, memory_key_mask=None, tgt_mask=None, tgt_key_mask=None):

        # adding embedding and position encoding.
        x = self.embedding(x.to(torch.long))  
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        #RESHAPE FOR PYTORCH: BxLxV -> LxBxV
        x = x.permute(1,0,2)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, encode_out, memory_mask=memory_mask, memory_key_padding_mask=memory_key_mask,
                                   tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_mask)
            
        #RESHAPE FOR OUTPUT: LxBxV -> BxLxV
        x = x.permute(1,0,2)
        return x
