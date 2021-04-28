# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
import torch

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
    AbstractiveSumXLNetDataset,
    XLNetDictionary,
)

from . import FairseqTask, register_task


@register_task('abstractive_summarization_xlnet')
class AbstractiveSummarizationXLNetTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--init-from-pretrained-doc-model', default='False', type=str, metavar='BOOL',
                            help='init model from a pretrained model')
        parser.add_argument('--pretrained-doc-model-path', default=None, metavar='PRETRAINED_PATH',
                            help='pretrained doc model path')
        # fmt: on

    # @staticmethod
    # def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
    #     model = utils.load_checkpoint_to_cpu(path)
    #     args = model['args']
    #     state_dict = model['model']
    #     args = utils.override_model_args(args, arg_overrides)
    #     src_dict = Dictionary.load(src_dict_path)
    #     tgt_dict = Dictionary.load(tgt_dict_path)
    #     assert src_dict.pad() == tgt_dict.pad()
    #     assert src_dict.eos() == tgt_dict.eos()
    #     assert src_dict.unk() == tgt_dict.unk()
    #
    #     task = AbstractiveSummarizationXLNetTask(args, src_dict, tgt_dict)
    #     model = task.build_model(args)
    #     model.upgrade_state_dict(state_dict)
    #     model.load_state_dict(state_dict, strict=True)
    #     return model

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = XLNetDictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        # src_eos_idx = src_dict.add_special_token('[END_OF_SENT]')
        # print('src_dict:[END_OF_SENT] id = {}, token = {}'.format(src_eos_idx, src_dict[src_eos_idx]))

        tgt_dict = XLNetDictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        # tgt_eos_idx = tgt_dict.add_special_token('[END_OF_SENT]')
        # print('tgt_dict:[END_OF_SENT] id = {}, token = {}'.format(tgt_eos_idx, tgt_dict[tgt_eos_idx]))

        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        assert src_dict.sep() == tgt_dict.sep()

        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = AbstractiveSumXLNetDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            append_eos_to_target=True,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def clear_cuda(self, sample):
        '''
        Copy from xxzhang
        :param sample:
        :return:
        '''
        src_size = sample['net_input']['src_tokens'].numel()
        tgt_size = sample['target'].numel()
        if src_size > self.max_src_size or tgt_size > self.max_tgt_size:
            torch.cuda.empty_cache()
            if src_size > self.max_src_size:
                self.max_src_size = src_size
            if tgt_size > self.max_tgt_size:
                self.max_tgt_size = tgt_size

    def load_pretrained_model(self, model, state_file_name):
        from torch.serialization import default_restore_location
        state = torch.load(state_file_name, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        params = state['model']

        non_encoder_param_names = [k for k in params.keys() if not k.startswith('encoder')]
        for nk in non_encoder_param_names:
            del params[nk]

        enc_cnt = 0
        non_enc_cnt = 0
        for k in params.keys():
            if not k.startswith('encoder'):
                print(k)
                non_enc_cnt += 1
            else:
                enc_cnt += 1
        print('enc_cnt = %d, non_enc_cnt = %d' % (enc_cnt, non_enc_cnt))
        model.load_state_dict(params, strict=False)
        print('*** *** load pretrained doc encoder done! *** ***')