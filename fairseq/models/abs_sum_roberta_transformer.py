# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import os
import json
import copy
import logging

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers import RobertaModel, RobertaConfig
# from pytorch_transformers import BertModel, BertConfig

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LayerNorm,
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
)


@register_model('roberta_transformer')
class AbsSumRobertaTransformerModel(FairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-dropout', type=float, metavar='D',
                            help='decoder dropout probability')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--layer-norm-eps', type=float, metavar='D',
                            help='eps for layer norm')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        # parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
        #                     help='comma separated list of adaptive softmax cutoff points. '
        #                          'Must be used with adaptive_loss criterion'),
        # parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
        #                     help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--roberta-model', default='roberta-base',
                            help="RoBerta pre-trained model selected in the list: roberta-base, "
                                 "roberta-large.")

        parser.add_argument('--roberta-decoder', default=False, action='store_true',
                            help='if set, the decoder is built as BERT architecture, instead of Fairseq transformer')
        parser.add_argument('--roberta-decoder-initialization', default=False, action='store_true',
                            help='if set, the decoder is built as BERT architecture, instead of Fairseq transformer')
        parser.add_argument('--roberta-config-path', default=None, metavar='PRETRAINED_PATH',
                            help='roberta config json file path')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)

        if hasattr(args, 'roberta_decoder') and args.roberta_decoder:
            print("Apply Bert Architecture as Decoder")
            # json_file_path = 'roberta-vocab/{0}-config.json'.format(args.roberta_model)
            json_file_path = args.roberta_config_path
            config = from_json_file(json_file_path)
            decoder_config = Namespace(**config)
            print(decoder_config)
            decoder = BertDecoder(args, decoder_config, tgt_dict, decoder_embed_tokens, left_pad=args.left_pad_target)
        else:
            decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=args.left_pad_target)
        return AbsSumRobertaTransformerModel(encoder, decoder)

    def forward(self, src_tokens, segment_ids, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, segment_ids)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out

    def initilize_roberta_decoder(self):
        print("Initializing the decoder with Roberta encoder parameters.")
        assert self.decoder is not None
        assert self.encoder is not None
        # Embedding
        # print(self.decoder.embeddings)
        # print(self.encoder.roberta.embeddings)
        self.decoder.embeddings = self.copy_params(self.encoder.roberta.embeddings, self.decoder.embeddings)

        # print(self.encoder.roberta.encoder.layer[0])
        # print(self.decoder.layers[0])
        # Layer list
        for i in range(len(self.encoder.roberta.encoder.layer)):
            self.decoder.layers[i] = self.copy_params(self.encoder.roberta.encoder.layer[i], self.decoder.layers[i])

    def copy_params(self, module1, module2):
        params1 = module1.state_dict()
        params2 = module2.state_dict()
        dict_param2 = dict(params2)
        for name1 in params1:
            # print(name1)
            # print(params1[name1].data)
            if name1 in dict_param2.keys():
                # print('before', dict_param2[name1])
                dict_param2[name1].data.copy_(params1[name1].data)
                # print('after', dict_param2[name1])
            # print('-------------------')
        module2.load_state_dict(dict_param2)
        return module2


def from_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, "r", encoding='utf-8') as reader:
        text = reader.read()
        json_object = json.loads(text)
        config = dict()
        for key, value in json_object.items():
            config[key] = value
    return config


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.n_gpu = torch.cuda.device_count()

        print('Distributed rank: ', args.distributed_rank)
        print('Number of used GPU: ', self.n_gpu)

        # if args.distributed_world_size > 1:
        #     if args.distributed_rank not in [-1, 0]:  # [1, 0]
        #         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        # Load pre-trained model (weights)
        config = RobertaConfig.from_pretrained(args.roberta_model)
        self.roberta = RobertaModel.from_pretrained(args.roberta_model, config=config)

        # if args.distributed_world_size > 1:
        #     if args.distributed_rank == 0:  # 1
        #         torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        # self.layers = nn.ModuleList([])
        # self.layers.extend([
        #     TransformerEncoderLayer(args)
        #     for i in range(args.encoder_layers)
        # ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    # def forward(self, src_tokens, src_lengths):
    def forward(self, src_tokens, segment_ids):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # print(src_tokens)
        # sum = src_tokens[:, 0].sum().item()
        # print(sum)
        bsz, seqlen = src_tokens.size()
        src_tokens = src_tokens.view(bsz, seqlen)
        segment_ids = segment_ids.view(bsz, seqlen)  # all fill 0

        # compute padding mask
        attention_mask = src_tokens.ne(self.padding_idx)
        # print(attention_mask)
        # enc_hids, _ = self.bert(src_tokens, segment_ids, attention_mask, output_all_encoded_layers=False)
        # print(src_tokens)
        enc_hids, _ = self.roberta(src_tokens, token_type_ids=segment_ids, attention_mask=attention_mask)

        # print('enc_hids', enc_hids.size())

        # doc_pos = self.sent_embed_positions(doc_pos_tok)

        # sent_repr = x[0].view(bsz, n_sent, -1)
        sent_repr = enc_hids

        # print( 'sent_repr', sent_repr.size() )

        if self.embed_positions is not None:
            sent_repr += self.embed_positions(src_tokens)

        # B x T x C -> T x B x C
        sent_repr = sent_repr.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        # for layer in self.layers:
        #     sent_repr = layer(sent_repr, encoder_padding_mask)

        if self.normalize:
            sent_repr = self.layer_norm(sent_repr)

        '''
                # embed tokens and positions
                x = self.embed_scale * self.embed_tokens(src_tokens)
                if self.embed_positions is not None:
                    x += self.embed_positions(src_tokens)
                x = F.dropout(x, p=self.dropout, training=self.training)

                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

                # compute padding mask
                encoder_padding_mask = src_tokens.eq(self.padding_idx)
                if not encoder_padding_mask.any():
                    encoder_padding_mask = None

                # encoder layers
                for layer in self.layers:
                    x = layer(x, encoder_padding_mask)

                if self.normalize:
                    x = self.layer_norm(x)

                return {
                    'encoder_out': x,  # T x B x C
                    'encoder_padding_mask': encoder_padding_mask,  # B x T
                }
                '''
        return {
            'encoder_out': sent_repr,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }




    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        left_pad (bool, optional): whether the input is left-padded
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.decoder_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # print(encoder_out)
        # print(incremental_state)
        # exit(1)
        # embed positions

        # incremental_state = None
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # self.project_in_dim = None
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers

        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # self.project_out_dim = None
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # self.adaptive_softmax = None
        # print(self.share_input_output_embed)
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.decoder_dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        # print("incremental_state", incremental_state) None
        # print("prev_attn_state", prev_attn_state) None
        # print("self_attn_mask", self_attn_mask.shape)  # tensor
        # print(self_attn_mask)
        # print("self_attn_padding_mask", self_attn_padding_mask) None
        # print("encoder_padding_mask", encoder_padding_mask) None

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)

        # print("prev_self_attn_state", prev_self_attn_state) None
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            # print("encoder_padding_mask", encoder_padding_mask)  # None
            # print(not self.training and self.need_attn)  # True
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m




@register_model_architecture('roberta_transformer', 'abs_sum_roberta_transformer_base')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.decoder_dropout = getattr(args, 'decoder_dropout', args.dropout)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('roberta_transformer', 'abs_sum_roberta_transformer')
def transformer_abs_sum_roberta(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    # args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    # args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    # args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    # args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)

@register_model_architecture('roberta_transformer', 'abs_sum_roberta_transformer_medium')
def transformer_abs_sum_roberta(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    # args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    # args.dropout = getattr(args, 'dropout', 0.15)
    base_architecture(args)

@register_model_architecture('roberta_transformer', 'abs_sum_roberta_transformer_large')
def transformer_abs_sum_roberta(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    # args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    # args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    # args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    # args.decoder_layers = getattr(args, 'decoder_layers', 6)
    # args.dropout = getattr(args, 'dropout', 0.15)
    base_architecture(args)

@register_model_architecture('roberta_transformer', 'abs_sum_roberta_large_transformer_large')
def transformer_abs_sum_roberta(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    # args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    # args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    # args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    # args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    # args.decoder_layers = getattr(args, 'decoder_layers', 12)
    # args.dropout = getattr(args, 'dropout', 0.15)
    base_architecture(args)

'''
@register_model_architecture('transformer', 'transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)
'''


###################################################################################################
### Bert as Decoder

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

# class BertConfig(PretrainedConfig):
#     r"""
#         :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
#         `BertModel`.
#
#
#         Arguments:
#             vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
#             hidden_size: Size of the encoder layers and the pooler layer.
#             num_hidden_layers: Number of hidden layers in the Transformer encoder.
#             num_attention_heads: Number of attention heads for each attention layer in
#                 the Transformer encoder.
#             intermediate_size: The size of the "intermediate" (i.e., feed-forward)
#                 layer in the Transformer encoder.
#             hidden_act: The non-linear activation function (function or string) in the
#                 encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
#             hidden_dropout_prob: The dropout probabilitiy for all fully connected
#                 layers in the embeddings, encoder, and pooler.
#             attention_probs_dropout_prob: The dropout ratio for the attention
#                 probabilities.
#             max_position_embeddings: The maximum sequence length that this model might
#                 ever be used with. Typically set this to something large just in case
#                 (e.g., 512 or 1024 or 2048).
#             type_vocab_size: The vocabulary size of the `token_type_ids` passed into
#                 `BertModel`.
#             initializer_range: The sttdev of the truncated_normal_initializer for
#                 initializing all weight matrices.
#             layer_norm_eps: The epsilon used by LayerNorm.
#     """
#     pretrained_config_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
#
#     def __init__(self,
#                  vocab_size_or_config_json_file=30522,
#                  hidden_size=768,
#                  num_hidden_layers=12,
#                  num_attention_heads=12,
#                  intermediate_size=3072,
#                  hidden_act="gelu",
#                  hidden_dropout_prob=0.1,
#                  attention_probs_dropout_prob=0.1,
#                  max_position_embeddings=512,
#                  type_vocab_size=2,
#                  initializer_range=0.02,
#                  layer_norm_eps=1e-12,
#                  **kwargs):
#         super(BertConfig, self).__init__(**kwargs)
#         # if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
#         #                 and isinstance(vocab_size_or_config_json_file, unicode)):
#         #     with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
#         #         json_config = json.loads(reader.read())
#         #     for key, value in json_config.items():
#         #         self.__dict__[key] = value
#         # elif isinstance(vocab_size_or_config_json_file, int):
#         self.vocab_size = vocab_size_or_config_json_file
#         self.hidden_size = hidden_size
#         self.num_hidden_layers = num_hidden_layers
#         self.num_attention_heads = num_attention_heads
#         self.hidden_act = hidden_act
#         self.intermediate_size = intermediate_size
#         self.hidden_dropout_prob = hidden_dropout_prob
#         self.attention_probs_dropout_prob = attention_probs_dropout_prob
#         self.max_position_embeddings = max_position_embeddings
#         self.type_vocab_size = type_vocab_size
#         self.initializer_range = initializer_range
#         self.layer_norm_eps = layer_norm_eps
#         # else:
#         #     raise ValueError("First argument must be either a vocabulary size (int)"
#         #                      "or the path to a pretrained model config file (str)")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# try:
#     from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except (ImportError, AttributeError) as e:
#     class BertLayerNorm(nn.Module):
#         def __init__(self, hidden_size, eps=1e-12):
#             """Construct a layernorm module in the TF style (epsilon inside the square root).
#             """
#             super(BertLayerNorm, self).__init__()
#             self.weight = nn.Parameter(torch.ones(hidden_size))
#             self.bias = nn.Parameter(torch.zeros(hidden_size))
#             self.variance_epsilon = eps
#
#         def forward(self, x):
#             u = x.mean(-1, keepdim=True)
#             s = (x - u).pow(2).mean(-1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#             return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = True  # config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x):

        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_hidden_states, key_hidden_states, value_hidden_states, attention_mask=None, head_mask=None):
        # print('query', query_hidden_states.shape)
        # print('key', key_hidden_states.shape)
        # print('value', value_hidden_states.shape)
        mixed_query_layer = self.query(query_hidden_states)
        mixed_key_layer = self.key(key_hidden_states)
        mixed_value_layer = self.value(value_hidden_states)
        # print('mixed_query_layer', mixed_query_layer.shape)
        # print('mixed_key_layer', mixed_key_layer.shape)
        # print('mixed_value_layer', mixed_value_layer.shape)

        tgt_len, bsz, embed_dim = query_hidden_states.size()

        # query_layer = self.transpose_for_scores(mixed_query_layer)
        # key_layer = self.transpose_for_scores(mixed_key_layer)
        # value_layer = self.transpose_for_scores(mixed_value_layer)
        query_layer = mixed_query_layer.contiguous().view(tgt_len, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        key_layer = mixed_key_layer.contiguous().view(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        value_layer = mixed_value_layer.contiguous().view(-1, bsz * self.num_attention_heads, self.attention_head_size).transpose(0, 1)

        # print('query_layer', query_layer.shape)
        # print('key_layer', key_layer.shape)
        # print('value_layer', value_layer.shape)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(1, 2))
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(0)
            # print('attention_scores', attention_scores.shape)
            # print('attention_mask', attention_mask.shape)
            attention_scores = attention_scores + attention_mask
            # attention_scores = attention_scores

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = utils.softmax(
            attention_scores, dim=-1
        ).type_as(attention_scores)
        attention_probs = F.dropout(attention_probs, p=self.attention_probs_dropout_prob, training=self.training)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
        # print('attention_probs', attention_probs.shape)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.bmm(attention_probs, value_layer)
        # context_layer = torch.matmul(attention_probs, value_layer)
        # print('attention_probs', attention_probs.shape)
        # print('value_layer', value_layer.shape)
        # print('context_layer', context_layer.shape)

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)
        # print('context layer', context_layer.shape)
        context_layer = context_layer.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # print('context layer', context_layer.shape)
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        # exit(1)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(query_hidden_states=query_tensor,
                                 key_hidden_states=key_tensor,
                                 value_hidden_states=value_tensor,
                                 attention_mask=attention_mask,
                                 head_mask=head_mask)
        attention_output = self.output(self_outputs[0], query_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class BertDecoderLayer(nn.Module):
    def __init__(self, config, args):
        super(BertDecoderLayer, self).__init__()
        self.attention = BertAttention(config)
        # self.self_intermediate = BertIntermediate(config)

        self.encoder_attention = MultiheadAttention(config.hidden_size, config.num_attention_heads,
            dropout=args.attention_dropout,)
        self.intermediate = BertIntermediate(config)

        self.output = BertOutput(config)

        self.need_attn = True

    def forward(self, x, encoder_hidden_states, encoder_padding_mask, self_attn_mask=None, head_mask=None):
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.


        self_attention_outputs = self.attention(query_tensor=x, key_tensor=x, value_tensor=x,
                                                     attention_mask=self_attn_mask, head_mask=head_mask)
        self_attention_output = self_attention_outputs[0]
        # self_intermediate_output = self.self_intermediate(self_attention_output)

        attention_outputs = self.encoder_attention(query=self_attention_output, key=encoder_hidden_states,
                                                   value=encoder_hidden_states, key_padding_mask=encoder_padding_mask,
                                                   incremental_state=None,
                                                   static_kv=True,
                                                   need_weights=(not self.training and self.need_attn),)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs



class BertDecoder(FairseqIncrementalDecoder):
    """
    Bert decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        left_pad (bool, optional): whether the input is left-padded
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, config, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)

        self.embeddings = BertEmbeddings(config)
        # self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = config.hidden_size  # embed_tokens.embedding_dim
        embed_dim = config.hidden_size  # args.decoder_embed_dim
        output_embed_dim = config.hidden_size  # args.decoder_output_dim

        # padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        # self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        # self.project_in_dim = BertLinear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        # self.embed_positions = BertPositionalEmbedding(
        #     args.max_target_positions, embed_dim, padding_idx,
        #     left_pad=left_pad,
        #     learned=args.decoder_learned_pos,
        # ) if not args.no_token_positional_embeddings else None
        self.embed_positions = None
        self.layers = nn.ModuleList([])
        self.layers.extend([
            BertDecoderLayer(config, args)
            for _ in range(config.num_hidden_layers)
        ])

        self.adaptive_softmax = None

        # self.project_out_dim = BertLinear(embed_dim, output_embed_dim, bias=False) \
        #     if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        # if args.adaptive_softmax_cutoff is not None:
        #     self.adaptive_softmax = AdaptiveSoftmax(
        #         len(dictionary),
        #         output_embed_dim,
        #         options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
        #         dropout=args.adaptive_softmax_dropout,
        #         adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
        #         factor=args.adaptive_softmax_factor,
        #         tie_proj=args.tie_adaptive_proj,
        #     )

        self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
        nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):

        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # print(encoder_out)
        # print(incremental_state)
        # exit(1)
        # embed positions

        # incremental_state = None
        # positions = self.embed_positions(
        #     prev_output_tokens,
        #     incremental_state=incremental_state,
        # ) if self.embed_positions is not None else None
        #
        # if incremental_state is not None:
        #     prev_output_tokens = prev_output_tokens[:, -1:]
        #     if positions is not None:
        #         positions = positions[:, -1:]

        # embed tokens and positions

        x = self.embeddings(prev_output_tokens)

        # if positions is not None:
        #     x += positions
        # x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        # print('new batch')
        # print(prev_output_tokens.shape)
        # print('x', x.shape)
        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            # print('=========')
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # self.project_out_dim = None
        # if self.project_out_dim is not None:
        #     x = self.project_out_dim(x)

        # self.adaptive_softmax = None
        # print(self.share_input_output_embed)
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            # if self.share_input_output_embed:
            #     x = F.linear(x, self.embed_tokens.weight)
            # else:
            x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict

def BertEmbedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def BertLinear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def BertPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


