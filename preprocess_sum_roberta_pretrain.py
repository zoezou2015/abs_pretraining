#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest

from fairseq import options, tasks
from fairseq.data import indexed_dataset, roberta_dictionary
from fairseq.binarizer import Binarizer
from fairseq.utils import import_user_module
from multiprocessing import Pool


import os
import shutil


def main(args):
    from fairseq import utils
    utils.xpprint(args)

    import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = roberta_dictionary.RobertaDictionary.load_json(args.srcdict)
            # src_dict.save('roberta-vocab/roberta-base-vocab.txt')
            print('load bert dict from {} | size {}'.format(args.srcdict, len(src_dict)))
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = roberta_dictionary.RobertaDictionary.load_json(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        print('input_prefix', input_prefix)
        print(dict_path(lang))

        dict = roberta_dictionary.RobertaDictionary.load(dict_path(lang))
        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        from pytorch_transformers import RobertaTokenizer
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        def penn_token2orig_token(sent):
            # -LRB- -RRB- -LSB- -RSB- -LCB- -RCB-
            penn2orig = {"``":'"', "''": '"',
                         "-LRB-": '(', "-RRB-": ')',
                         "-LSB-":'[', "-RSB-":']',
                         "-LCB-":'{', "-RCB-":'}'}
            words = sent.strip().split()
            words = [wd if not wd in penn2orig else penn2orig[wd] for wd in words]
            return ' '.join(words)

        num_token, num_unk_token = 0, 0
        num_seq = 0
        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )
        output_ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, 'article_next', "bin")
        )
        truncated_number = 512
        output_length = 256

        CLS_TOKEN = '<s>'
        SEP_TOKEN = '</s>'

        for line in open(input_file, encoding='utf8'):
            sents = line.strip().split('<S_SEP>')
            sents = [tokenizer.tokenize(penn_token2orig_token(sent)) for sent in sents]
            article_toks = []
            for i, sent in enumerate(sents):
                if i != 0:
                    article_toks.append(SEP_TOKEN)
                article_toks.extend(sent)
            article_segments = []
            output_segments = []
            tmp_seg = []
            for i, tok in enumerate(article_toks):
                if len(tmp_seg) == 0:
                    tmp_seg.append(CLS_TOKEN)
                tmp_seg.append(tok)
                if tok == SEP_TOKEN:
                    tmp_seg.append(tok)
                if len(tmp_seg) >= truncated_number:
                    tmp_seg = tmp_seg[:truncated_number]
                    if tmp_seg[-1] != SEP_TOKEN:
                        tmp_seg[-1] = SEP_TOKEN
                    tmp_output = article_toks[i+1: min(i+1+output_length, len(article_toks))]
                    if len(tmp_output) < 0.3*output_length:
                        break
                    article_segments.append(tokenizer.convert_tokens_to_ids(tmp_seg))
                    output_segments.append(tokenizer.convert_tokens_to_ids(tmp_output))
                    tmp_seg = []
            assert len(article_segments) == len(output_segments)
            for i in range(len(article_segments)):
                assert len(article_segments[i]) <= truncated_number
                assert len(output_segments[i]) <= output_length and len(output_segments[i]) >= 0.3*output_length
                tensor = torch.IntTensor(article_segments[i])
                ds.add_item(tensor)
                output_tensor = torch.IntTensor(output_segments[i])
                output_ds.add_item(output_tensor)

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
        output_ds.finalize(dataset_dest_file(args, output_prefix, 'article_next', "idx"))
        print('done!')
        # print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
        #     lang, input_file, num_seq, num_token,
        #     100 * num_unk_token / num_token, dict.unk_word if hasattr(dict, 'unk_word') else '<no_unk_word>'))

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format == "binary":
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(lang, vocab):
        if args.trainpref:
            print(args.trainpref, lang)
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers)
        # if args.testpref:
        #     for k, testpref in enumerate(args.testpref.split(",")):
        #         outprefix = "test{}".format(k) if k > 0 else "test"
        #         make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    make_all(args.source_lang, src_dict)
    # if target:
    #     make_all(args.target_lang, tgt_dict)

    print("| Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args.destdir,
                    "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def merge_files(files, outpath):
    ds = indexed_dataset.IndexedDatasetBuilder("{}.bin".format(outpath))
    for file in files:
        ds.merge_file_(file)
        os.remove(indexed_dataset.data_file_path(file))
        os.remove(indexed_dataset.index_file_path(file))
    ds.finalize("{}.idx".format(outpath))


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
