#!/usr/bin/env python3

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import linear, softmax, dropout
from torch import Tensor
from torch.nn import Module, Parameter, Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.container import ModuleList
import copy

from fairseq.models import FairseqEncoder
from fairseq.models.speech_recognition.transformer_encoder import SRTransformerEncoder
from fairseq.models.speech_recognition.convsubsampler import Conv1dSubsampler, Pooling1DSubsampler, SuperFrame
from fairseq.data.data_utils import lengths_to_padding_mask

def mine_softmax(x, dim=0):
    x.clamp_(torch.finfo(x.dtype).min)
    return softmax(x, dim=dim)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

# use positional embedding, which contains a learning embedding layer
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.maxlen  = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model)
        self.pe_v = torch.nn.Embedding(2*maxlen, d_model)

    # out dim: L x L x d'  (d' = d_model // nhead)
    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen-1)
        # nn.Embedding layer is used, need to be positive
        pos_seq = pos_seq + self.maxlen
        return self.pe_k(pos_seq), self.pe_v(pos_seq)

    def forward_abs(self, pos_seq):
        pos_seq.clamp_(0, 2*self.maxlen-1)
        return self.pe_k(pos_seq), self.pe_v(pos_seq)

# use fix positional encoding
class PositionalEncodding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncodding, self).__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, "d_model in the position encoding requires dimension to be even, as it needs to add sin and cos together"
        inv_freq = 1/ (10000 ** (torch.arange(0.0, d_model, 2.0)/d_model))
        self.register_buffer('inv_freq', inv_freq)

    # out dim: L x L x d'  (d' = d_model // nhead)
    def forward(self, pos_seq):
        pos_seq = pos_seq.type_as(self.inv_freq)
        sinusoid_inp = torch.ger(pos_seq.view(-1), self.inv_freq).view(pos_seq.size(0), pos_seq.size(1), -1)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb, pos_emb

    def forward_v2(self, pos_seq):
        pos_seq = pos_seq.float()
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb

class MultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        assert self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.out_proj_linear = Linear(embed_dim, embed_dim, bias=bias)

        # self.in_proj_linear = Linear(3*embed_dim, embed_dim)
        self.in_proj_linear_k = Linear(embed_dim, embed_dim)
        self.in_proj_linear_q = Linear(embed_dim, embed_dim)
        self.in_proj_linear_v = Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_linear_k.weight.data)
        constant_(self.in_proj_linear_k.bias.data, 0.)
        xavier_uniform_(self.in_proj_linear_q.weight.data)
        constant_(self.in_proj_linear_q.bias.data, 0.)
        xavier_uniform_(self.in_proj_linear_v.weight.data)
        constant_(self.in_proj_linear_v.bias.data, 0.)
        xavier_uniform_(self.out_proj_linear.weight.data)
        constant_(self.out_proj_linear.bias.data, 0.)


        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, pos_k, pos_v, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """

        embed_dim_to_check = self.embed_dim
        num_heads = self.num_heads
        bias_k = self.bias_k
        bias_v = self.bias_v
        add_zero_attn = self.add_zero_attn
        dropout_p   = self.dropout
        training = self.training
        static_k = None
        static_v = None

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        if key is not None and value is not None:
            assert key.size() == value.size()

        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q = self.in_proj_linear_q(query)
        k = self.in_proj_linear_k(key)
        v = self.in_proj_linear_v(value)

        q = q * scaling

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        if bias_k is not None and bias_v is not None:
            if static_k is None and static_v is None:
                k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))
            else:
                assert static_k is None, "bias cannot be added to static key."
                assert static_v is None, "bias cannot be added to static value."
        else:
            assert bias_k is None
            assert bias_v is None

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if static_k is not None:
            assert static_k.size(0) == bsz * num_heads
            assert static_k.size(2) == head_dim
            k = static_k

        if static_v is not None:
            assert static_v.size(0) == bsz * num_heads
            assert static_v.size(2) == head_dim
            v = static_v

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if add_zero_attn:
            src_len += 1
            k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.nn.functional.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = torch.nn.functional.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        # add position embedding here
        # q dim: B*nhead x L x d'    (d' = d_model//nhead)
        # pos_k dim: L x L x d'
        # attn_output_weight dim: B*nhead x L x L
        if pos_k is not None:
            B = torch.bmm(q.transpose(0, 1), pos_k.transpose(1, 2)).transpose(0, 1)
            attn_output_weights = attn_output_weights + B

        if attn_mask is not None:
            attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        # attn_output_weights = softmax(attn_output_weights, dim=-1)
        attn_output_weights = mine_softmax(attn_output_weights, dim=-1)

        attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

        attn_output = torch.bmm(attn_output_weights, v)

        # attn_output_weight dim: B*nhead x L x L
        # pos_v dim: L x L x d'
        # attn_output = B*nhead x L x d'
        if pos_v is not None:
            attn_v = torch.bmm(attn_output_weights.transpose(0,1), pos_v).transpose(0,1)
            attn_output = attn_output + attn_v


        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = self.out_proj_linear(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None

class ConvModule(nn.Module):
    def __init__(self, input_dim, kernel_size, dropout_rate, causal=True, bn=False):
        super(ConvModule, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

        self.pw_conv_1 = nn.Conv2d(1, 2, 1, 1, 0)
        self.glu_act = torch.nn.Sigmoid()
        self.causal = causal
        self.bn = bn
        self.kernel_size = kernel_size
        if causal:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1), groups=input_dim)
        else:
            self.dw_conv_1d = nn.Conv1d(input_dim, input_dim, kernel_size, 1, padding=(kernel_size-1)//2, groups=input_dim)
        if bn:
            self.BN = nn.BatchNorm1d(input_dim)
        self.act = nn.ReLU()
        self.pw_conv_2 = nn.Conv2d(1, 1, 1, 1, 0)
        self.dropout = nn.Dropout(dropout_rate)

        self.pw_conv_simplify_w = torch.nn.Parameter(torch.ones(3))
        self.pw_conv_simplify_b = torch.nn.Parameter(torch.zeros(3))
        self.flag_pw_conv_simplify = True

    def forward(self, x):
        # input dim: L x b x D
        # need to permute to b x L x D for CNN
        x = x.permute(1, 0, 2)
        x = self.layer_norm(x)
        if self.flag_pw_conv_simplify:
            x_0 = x * self.pw_conv_simplify_w[0] + self.pw_conv_simplify_b[0]
            x_1 = x * self.pw_conv_simplify_w[1] + self.pw_conv_simplify_b[1]
            x = x_0 + x_1
            # x = x_0 * self.glu_act(x_1)
        else:
            x = x.unsqueeze(1)
            x = self.pw_conv_1(x)
            x = x[:, 0] * self.glu_act(x[:, 1])
        x = x.permute([0, 2, 1])
        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, :-(self.kernel_size-1)]
        '''
        if self.bn:
            x = self.BN(x)
        '''
        x = self.act(x)
        x = x.unsqueeze(1).permute([0, 1, 3, 2])
        if self.flag_pw_conv_simplify:
            x = x * self.pw_conv_simplify_w[2] + self.pw_conv_simplify_b[2]
        else:
            x = self.pw_conv_2(x)
        x = self.dropout(x).squeeze(1)
        # x need to permuete back to L x b x D for the following transfomer operation
        x = x.permute(1, 0, 2)

        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout_rate):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout_rate)
        )


    def forward(self, x):
        x = self.layer_norm(x)
        out = self.net(x)

        return out


class ConformerLayer(nn.Module):
    """Encoder layer module.
    https://arxiv.org/abs/2005.08100
    :param int d_model: attention hidden dim
    :param int nhead: the number of multi-head attention
    :param int dim_feedforward: feed_forward dim
    :param int kernel_size: kernel_size of cnn module
    :param float dropout: dropout ratio
    :param bool causal: causal cnn is needed for streaming model
    :param bool bn: batchnorm

    """

    def __init__(self, d_model, nhead=8, dim_feedforward=2048, kernel_size=3, dropout=0.1, causal=True ,bn=False, activation='gelu'):
        """Construct an EncoderLayer object."""
        super(ConformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead   = nhead
        self.feed_forward_in = FeedForward(d_model, dim_feedforward, dropout)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.conv = ConvModule(d_model, kernel_size, dropout, causal,bn)
        self.feed_forward_out = FeedForward(d_model, dim_feedforward, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_k, pos_v, src_mask=None, src_key_padding_mask=None):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = x + 0.5 * self.feed_forward_in(x)
        x = x + self.self_attn(x, x, x, pos_k, pos_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out


class ChunkTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", pre_norm=False, attention_dropout=0.0):
        super(ChunkTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead   = nhead
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=attention_dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.norm_k = LayerNorm(d_model // nhead)
        self.norm_v = LayerNorm(d_model // nhead)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(ChunkTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, pos_k, pos_v, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if self.pre_norm:
            residual = src
            src = self.norm1(src)
            '''
            if pos_k is not None:
                pos_k = self.norm_k(pos_k)
            if pos_v is not None:
                pos_v = self.norm_v(pos_v)
            '''
            src2 = self.self_attn(src, src, src, pos_k, pos_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = residual + self.dropout1(src2)

            residual = src
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = residual + self.dropout2(src2)

        else:
            src2 = self.self_attn(src, src, src, pos_k, pos_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src

class SRChunkTransformerEncoder(FairseqEncoder):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    # def __init__(self, encoder_layer, num_layers, norm=None, pos_emb_type="rel", use_pos_encodding=False, pre_norm=False):
    def __init__(self, args):
        super().__init__(None)

        if args.subsample_type == "superframe":
            self.subsample = SuperFrame(args.encoder_embed_dim)
        else:
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )

        # transformer or conformer
        # hyper-parameter for encoder layer
        self.encoder_type = args.encoder_type
        self.hidden_size  = args.encoder_embed_dim
        self.num_head = args.encoder_attention_heads
        self.dim_feedforward = args.encoder_ffn_embed_dim
        self.dropout = args.dropout
        self.activation = args.activation_fn
        self.pre_norm = args.encoder_normalize_before
        self.attention_dropout = args.attention_dropout

        # hyper-parameter for encoder
        self.num_layers = args.encoder_layers
        self.norm = torch.nn.LayerNorm(self.hidden_size)
        self.bidirectional = args.bidirectional
        self.n_future_chunksize = args.n_future_chunksize
        self.n_prevhist = args.n_prevhist

        # position embedding/encoding
        self.pos_emb_type = args.pos_enc_type   # rel or abs or none
        self.use_pos_encodding = args.use_pos_encodding     # encoding or embedding
        # use learnable position embedding
        self.pos_emb = PositionalEmbedding(self.hidden_size//self.num_head, 1000)
        # use fix position encoding
        self.pos_enc = PositionalEncodding(self.hidden_size//self.num_head)

        if self.encoder_type == "chunktransformer":
            encoder_layer = ChunkTransformerEncoderLayer(self.hidden_size, nhead=self.num_head,
                                                         dim_feedforward=self.dim_feedforward, dropout=self.dropout,
                                                         activation=self.activation, pre_norm=self.pre_norm,
                                                         attention_dropout=self.attention_dropout)
        elif self.encoder_type == "chunkconformer":
            encoder_layer = ConformerLayer(self.hidden_size, nhead=self.num_head,
                                           dim_feedforward=self.dim_feedforward, dropout=self.dropout,
                                           activation='gelu')
        self.layers = _get_clones(encoder_layer, self.num_layers)


    def gen_chunk_mask(self, x_len, future_chunksize, hist_len=0):

        # at least see the current frame
        if future_chunksize < 1:
            future_chunksize = 1

        chunk_start_indices = list(range(0, x_len, future_chunksize))
        # the last element must the x_len, start idx is 0 by default
        chunk_start_indices.append(x_len)
        chunk_start_indices.remove(0)
        mask = torch.BoolTensor(x_len, x_len).fill_(False)

        prev_idx = 0
        if hist_len > 0:
            for idx in chunk_start_indices:
                mask[prev_idx:idx, max(0, prev_idx-hist_len):idx] = True
                prev_idx = idx
        else:
            for idx in chunk_start_indices:
                mask[prev_idx:idx, 0:idx] = True
                prev_idx = idx

        init_mask = torch.zeros(x_len, x_len).masked_fill(mask, 1.0)
        attn_mask = init_mask.cuda()
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
        return attn_mask

    def gen_pos_embedding(self, src):
        if self.pos_emb_type == 'rel_pos':
            x_len = src.size(0)
            pos_seq = torch.arange(0, x_len).long().to(src.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]

            # use fix encoding, instead of learnable embedding for position
            if self.use_pos_encodding is True:
                pos_k, pos_v = self.pos_enc(pos_seq)
            else:
                pos_k, pos_v = self.pos_emb(pos_seq)
        elif self.pos_emb_type == 'abs_pos':
            x_len = src.size(0)
            pos_seq = torch.arange(0, x_len).long().to(src.device).unsqueeze(0).expand(x_len, x_len)
            # use fix encoding, instead of learnable embedding for position
            if self.use_pos_encodding is True:
                pos_k, pos_v = self.pos_enc(pos_seq)
            else:
                pos_k, pos_v = self.pos_emb.forward_abs(pos_seq)
        else:
            pos_k, pos_v = None, None
        return pos_k, pos_v


    # def forward(self, src, mask=None, src_key_padding_mask=None):
    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        """
        Args:
            src_tokens: Input source tokens Tensor of shape B X T X C
            src_lengths: Lengths Tensor corresponding to input source tokens
            return_all_hiddens: If true will append the self attention states to the encoder states
        Returns:
            encoder_out: Tensor of shape B X T X C
            encoder_padding_mask: Optional Tensor with mask
            encoder_embedding: Optional Tensor. Always empty here
            encoder_states: List of Optional Tensors wih self attention states
            src_tokens: Optional Tensor. Always empty here
            src_lengths: Optional Tensor. Always empty here
        """

        # return x shape: T X B X C
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        # x dim: B x T x D
        # output = x

        if self.bidirectional:
            attn_mask = None
        else:
            attn_mask = self.gen_chunk_mask(x.size(0), self.n_future_chunksize, self.n_prevhist)

        # need to use key_padding_mask for lc-transformer, no need for uni-transformer
        key_padding_mask = torch.arange(0, x.size(0), 1, device=x.device, dtype=torch.int64)
        key_padding_mask = key_padding_mask.unsqueeze(0).expand(x.size(1), -1)
        key_padding_mask = key_padding_mask.ge(input_lengths.unsqueeze(1))

        # out = out.permute(1, 0, 2)

        # x dim: T x B x C
        pos_k, pos_v = self.gen_pos_embedding(x)


        encoder_states = []
        for l, mod in enumerate(self.layers):
            x = mod(x, pos_k, pos_v, src_mask=attn_mask, src_key_padding_mask=key_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.norm is not None:
            x = self.norm(x)

        # x = x.permute(1, 0, 2)

        # return output

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_out_lengths": [input_lengths],
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """Required method for a FairseqEncoder. Calls the method from the parent class"""
        return SRTransformerEncoder.reorder_encoder_out(self, encoder_out, new_order)

