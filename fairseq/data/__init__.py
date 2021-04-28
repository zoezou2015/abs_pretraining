# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .dictionary import Dictionary, TruncatedDictionary
from .fairseq_dataset import FairseqDataset
from .backtranslation_dataset import BacktranslationDataset
from .concat_dataset import ConcatDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset
from .language_pair_dataset import LanguagePairDataset
from .lm_context_window_dataset import LMContextWindowDataset
from .monolingual_dataset import MonolingualDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .abs_sum_bert_dataset import AbstractiveSumBertDataset
from .abs_sum_xlnet_dataset import AbstractiveSumXLNetDataset
from .abs_sum_roberta_dataset import AbstractiveSumRobertaDataset
from .bert_dictionary import BertDictionary
from .xlnet_dictionary import XLNetDictionary
from .roberta_dictionary import RobertaDictionary

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'BacktranslationDataset',
    'ConcatDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'LMContextWindowDataset',
    'MonolingualDataset',
    'RoundRobinZipDatasets',
    'ShardedIterator',
    'TokenBlockDataset',
    'TransformEosDataset',
    'AbstractiveSumBertDataset',
    'AbstractiveSumXLNetDataset',
    'AbstractiveSumRobertaDataset',
    'BertDictionary',
    'XLNetDictionary',
    'RobertaDictionary',

]
