import logging
import torch

from fairseq.models.speech_recognition.convsubsampler import Conv1dSubsampler, Pooling1DSubsampler, SuperFrame
from fairseq.models.speech_recognition.transformer_encoder import SRTransformerEncoder
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules.conformer_layer import ConformerEncoderLayer
from fairseq.models import FairseqEncoder
from fairseq.modules import PositionalEmbedding, RelPositionalEncoding
import math

logger = logging.getLogger(__name__)

class SRConformerEncoder(FairseqEncoder):
    """Conformer Encoder for speech translation based on https://arxiv.org/abs/2005.08100"""

    def __init__(self, args):
        super().__init__(None)
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        if args.subsample_type == "superframe":
            self.subsample = SuperFrame(args.encoder_embed_dim)
        else:
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")],
            )

        self.pos_enc_type = args.pos_enc_type
        if self.pos_enc_type == "rel_pos":
            self.embed_positions = RelPositionalEncoding(
                args.max_source_positions, args.encoder_embed_dim
            )
        elif self.pos_enc_type == "rope":
            self.embed_positions = None
        else:  # Use absolute positional embedding
            self.pos_enc_type = "abs"
            self.embed_positions = PositionalEmbedding(
                args.max_source_positions, args.encoder_embed_dim, self.padding_idx
            )

        self.linear = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        self.dropout = torch.nn.Dropout(args.dropout)
        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    embed_dim=args.encoder_embed_dim,
                    ffn_embed_dim=args.encoder_ffn_embed_dim,
                    attention_heads=args.encoder_attention_heads,
                    dropout=args.dropout,
                    depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,
                    attn_type=args.attn_type,
                    pos_enc_type=self.pos_enc_type,
                    use_fp16=args.fp16,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        # self.encoder_out_poollayer = Pooling1DSubsampler(3, 3, 0)
        self.use_encoder_output_subsampler = args.use_encoder_output_subsampler
        if self.use_encoder_output_subsampler:
            self.encoder_out_poollayer = Pooling1DSubsampler(args.pool_kernel_size, args.pool_stride_size, args.pool_padding_size)

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
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
        x, input_lengths = self.subsample(src_tokens, src_lengths)  # returns T X B X C

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        x = self.embed_scale * x
        if self.pos_enc_type == "rel_pos":
            positions = self.embed_positions(x)

        elif self.pos_enc_type == "rope":
            positions = None

        else:
            positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
            x += positions
            positions = None

        x = self.linear(x)
        x = self.dropout(x)
        encoder_states = []

        # x is T X B X C
        for layer in self.conformer_layers:
            x, _ = layer(x, encoder_padding_mask, positions)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.use_encoder_output_subsampler:
            x, input_lengths = self.encoder_out_poollayer(x, input_lengths)

        # x = x[1::2, :, :]
        # input_lengths = input_lengths // 2
        # encoder_padding_mask = lengths_to_padding_mask(input_lengths)

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

