import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
from transformer.Layers import FFTBlock
from text.symbols import symbols
from mamba_ssm import Mamba2

import hparams as hp

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    
    return torch.FloatTensor(sinusoid_table)


class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_src_vocab=len(symbols) + 1,
                 len_max_seq=hp.max_seq_len,
                 d_word_vec=hp.encoder_hidden,
                 d_model=hp.encoder_hidden,
                 d_state=128,  # State-space model hidden size
                 d_conv=3,  # Convolution kernel size
                 n_layers=hp.encoder_layer,
                 dropout=hp.encoder_dropout):
        
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(len_max_seq + 1, d_word_vec).unsqueeze(0), requires_grad=False)

        # Replace FFTBlock with Mamba2 blocks
        self.layer_stack = nn.ModuleList([Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=d_model // 2,  # Headdim based on head count, adjust if necessary
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, mask, return_attns=False):

        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        
        # Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)

        return enc_output



class Decoder(nn.Module):
    ''' Decoder '''

    def __init__(self,
                 len_max_seq=hp.max_seq_len,
                 d_word_vec=hp.decoder_hidden,
                 d_model=hp.decoder_hidden,
                 d_state=128,  # State-space model hidden size
                 d_conv=3,  # Convolution kernel size
                 n_layers=hp.decoder_layer,
                 dropout=hp.decoder_dropout):
        
        super(Decoder, self).__init__()

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(len_max_seq + 1, d_word_vec).unsqueeze(0), requires_grad=False)

        # Replace FFTBlock with Mamba2 blocks
        self.layer_stack = nn.ModuleList([Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            headdim=d_model // 2,  # Headdim based on head count, adjust if necessary
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, mask, return_attns=False):

        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        
        # Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # Forward
        dec_output = enc_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output)

        return dec_output
