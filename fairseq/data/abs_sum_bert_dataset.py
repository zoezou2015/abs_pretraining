# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import random

from fairseq import utils

from . import data_utils, FairseqDataset


# def list2tokens(cls_idx, lst, sep_idx, pad_idx):
#     '''
#     padding [cls] w1 w2 ... [sep]
#     :param cls_idx:
#     :param lst:
#     :param sep_idx:
#     :return:
#     '''
#     tokens = [cls_idx]
#     first = True
#     for i, v in enumerate(lst):
#         tokens.append(sep_idx)
#         first = False
#         continue
#         tokens.append(v.item())
#     tokens.append(sep_idx)
#     return tokens
#
# def create_src_tok_batch(samples, cls_idx, pad_idx, sep_idx):
#     docs = []
#     max_src_length = -1
#     for s in samples:
#         src = s['source']
#         src_doc = list2tokens(cls_idx, src, sep_idx, pad_idx)
#         max_src_length = len(src_doc) if max_src_length < len(src_doc) else max_src_length
#         docs.append(src_doc)
#     print('max_src_length', max_src_length)
#     bsz = len(docs)
#
#


# def split_list(lst, key):
#     istart = 0
#     res = []
#     sublist = []
#     for i, v in enumerate(lst):
#         sublist.append(v.item())
#         if v == key:
#             if len(sublist) > 0:
#                 res.append( sublist )
#             sublist = []
#     if len(sublist) > 0:
#         res.append(sublist)
#
#     return res
#
# # right padding [cls] w1 w2 ... [sep]
# def docs2tensor(docs, max_nsent, max_sent_len, pad_idx, sep_idx):
#     bsz = len(docs)
#     src_tokens = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(pad_idx)
#     segment_ids = torch.LongTensor(bsz, max_nsent, max_sent_len).fill_(0)
#     src_tokens[:, :, 0] = docs[0][0][0]
#     doc_pad_mask = torch.ByteTensor(bsz, max_nsent).fill_(1)
#     for i, doc in enumerate(docs):
#         for j, sent in enumerate(doc):
#             doc_pad_mask[i, j] = 0
#             sent_len = len(sent)
#             src_tokens[i, j, 0: sent_len] = torch.LongTensor(sent)
#
#     return src_tokens, doc_pad_mask, segment_ids
#
# def create_src_tok_batch(samples, sep_id, cls_idx, pad_idx, max_sent_length=None):
#     docs = []
#     max_nsent = 0
#     max_sent_len = 0
#     for sample in samples:
#         src = sample['source']
#         sents = split_list(src, sep_id)
#
#         # if max_sent_length is not None:
#         #     sents = [sent if len(sent) <= max_sent_length else sent[0:max_sent_length] for sent in sents]
#         sents = [[cls_idx] + sent for sent in sents]
#
#         if sents[-1][-1] != sep_id:
#             sents[-1].append(sep_id)
#         max_nsent = max(max_nsent, len(sents))
#         cur_max_sent_len = max( map(len, sents) )
#         max_sent_len = max( max_sent_len, cur_max_sent_len )
#         docs.append(sents)
#
#     return docs2tensor(docs, max_nsent, max_sent_len, pad_idx, sep_id)


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    # create_src_tok_batch(samples, cls_idx, pad_idx, sep_idx)

    src_tokens = merge('source', left_pad=left_pad_source)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    segment_ids = src_tokens.clone()
    segment_ids.data.fill_(0)


    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'segment_ids': segment_ids,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class AbstractiveSumBertDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        pretrained_task=None, mask_ratio=None, max_mask_len=None, min_mask_len=None,

    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.sep() == tgt_dict.sep()
            assert src_dict.unk() == tgt_dict.unk()
            # assert src_dict.eos() == tgt_dict.eos()

        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target

        self.pretrained_task = pretrained_task
        self.training = True
        self.mask_ratio = mask_ratio
        self.max_mask_len = max_mask_len
        self.min_mask_len = min_mask_len

    def __getitem__(self, index):
        if self.pretrained_task == 'masked_span':
            src_tensor = self.src[index]
            src_list = src_tensor.tolist()  # [self.src_dict.eos_index] + src_item.tolist()
            # print()
            # org_src = [self.src_dict[word] for word in src_list]
            # print('orig src: ', org_src)

            first_pad_pos = -1
            for idx in src_list:
                if idx == self.src_dict.pad_index:
                    first_pad_pos = idx
                    break
            start, end = self.mask_interval(len(src_list), first_pad_pos)  # last [SEP] position
            target = src_list[start: end].copy()

            # mask source sentence
            source = []
            for i, w in enumerate(src_list):
                if i >= start and i < end:
                    w = self.mask_word(w)
                if w is not None:
                    source.append(w)
            assert len(src_list) == len(source)
            assert len(source) <= self.max_source_positions
            assert len(target) <= self.max_target_positions
            # new_src = [self.src_dict[word] for word in source]
            # print('new src: ', new_src)
            # new_tgt = [self.src_dict[word] for word in target]
            # print('new tgt: ', new_tgt)
            src_item = torch.LongTensor(source)
            tgt_item = torch.LongTensor(target)
        elif self.pretrained_task == 'mix_order':
            src_item = self.src[index]
            src_list = src_item.tolist()[1:-1]  # [self.src_dict.eos_index] + src_item.tolist()
            tokens = [self.src_dict[idx] for idx in src_list]
            src_tokens = ' '.join(tokens).split(self.src_dict.sep_word)
            sents = []
            labels = []
            for i, tokens in enumerate(src_tokens):
                sents.append([self.src_dict.index(tok) for tok in tokens.split(' ')])
                labels.append(i)
            # print()
            random.shuffle(labels)
            tgt_list = []
            for l in labels:
                tgt_list.extend(sents[l])
            # print(labels)
            # print(self.src_dict.sep_index)
            # print(sents)
            tgt_item = torch.LongTensor(tgt_list)
        else:
            tgt_item = self.tgt[index] if self.tgt is not None else None
            src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def mask_interval(self, len, first_pad_pos):
        if first_pad_pos > 0:
            len = min(len, first_pad_pos)
        start = np.random.randint(1, max(2, (len-self.min_mask_len)))
        mask_length = np.random.randint(self.min_mask_len, self.max_mask_len)
        return start, min(len-1, start + mask_length)

    def mask_word(self, w):
        p = np.random.random()
        if p < 0.8:  # 80%
            return self.src_dict.mask_index
        elif p < 0.5:  # 10%
            return self.get_random_word()
        else:  # 10%
            return w

    def get_random_word(self):
        rand_tok = self.src_dict.mask_index
        while rand_tok in self.special_idx:
            rand_tok = np.random.randint(1, len(self.src_dict) - 1)
        return rand_tok


    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, batch_size=None, src_len=10240, tgt_len=10240):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = batch_size if batch_size else max(num_tokens // max(src_len, tgt_len), 1)
        print('dummy batch: ntokens = {}, src_len = {}, tgt_len = {}, bsz = {}'.format(num_tokens, src_len, tgt_len, bsz))


        # def create_src(src_len):
        #     sent_len = 50
        #     num_sent = src_len // sent_len
        #     last_sent_len = src_len - sent_len * num_sent
        #     doc = [self.src_dict.roberta_cls()]
        #     for i in range(num_sent):
        #         if i != 0:
        #             doc.append(self.src_dict.sep())
        #         for j in range(sent_len-2):
        #             doc.append(self.src_dict.unk())
        #         doc.append(self.src_dict.sep())
        #     for i in range(last_sent_len):
        #         if i == 0:
        #             doc.append(self.src_dict.sep())
        #         doc.append(self.src_dict.unk())
        #     doc[-1] = self.src_dict.sep()
        #     return torch.LongTensor(doc)
        #
        # def create_tgt(tgt_len):
        #     sent_len = 50
        #     num_sent = tgt_len // sent_len
        #     last_sent_len = tgt_len - sent_len * num_sent
        #     summary = []
        #     for i in range(num_sent):
        #         if i != 0:
        #             summary.append(self.tgt_dict.sep())
        #         for j in range(sent_len-1):
        #             summary.append(self.tgt_dict.unk())
        #     for i in range(last_sent_len):
        #         if i == 0:
        #             summary.append(self.tgt_dict.sep())
        #         summary.append(self.tgt_dict.unk())
        #     return torch.LongTensor(summary)

        return self.collater([
            {
                'id': i,
                'source':  self.src_dict.dummy_sentence(src_len),
                'target':  self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)
