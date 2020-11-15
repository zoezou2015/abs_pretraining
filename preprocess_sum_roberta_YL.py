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

        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        def penn_token2orig_token(sent):
            # -LRB- -RRB- -LSB- -RSB- -LCB- -RCB-
            penn2orig = {"``":'"', "''": '"',
                         "-LRB-": '(', "-RRB-": ')',
                         "-LSB-":'[', "-RSB-":']',
                         "-LCB-":'{', "-RCB-":'}',
                         "-lrb-": '(', "-rrb-": ')',
                         "-lsb-": '[', "-rsb-": ']',
                         "-lcb-": '{', "-rcb-": '}',
                         }
            words = sent.strip().split()
            words = [wd if not wd in penn2orig else penn2orig[wd] for wd in words]
            return ' '.join(words)

        num_token, num_unk_token = 0, 0
        num_seq = 0
        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )
        truncated_number = 512 if lang == 'article' else 256
        CLS_TOKEN = '<s>'
        SEP_TOKEN = '</s>'
        if lang == 'article':
            for line in open(input_file, encoding='utf8'):
                article_wids = []
                min_src_sentence = 3
                max_src_sentence = 100
                max_src_ntokens_per_sent = 200
                min_src_ntokens_per_sent = 5
                sents = line.strip().split('<S_SEP>')
                sents = [sent.strip().split() for sent in sents]
                idxs = [i for i, sent in enumerate(sents) if (len(sent) > min_src_ntokens_per_sent)]
                src = [sents[i][:max_src_ntokens_per_sent] for i in idxs]
                src = src[:max_src_sentence]
                src_txt = [' '.join(sent) for sent in src]
                src_tokens = [tokenizer.tokenize(sent) for sent in src_txt]
                for i, sent in enumerate(src_tokens):
                    MAX_SENT_NTOKENS = 500
                    if len(sent) > MAX_SENT_NTOKENS:
                        sent = sent[:MAX_SENT_NTOKENS]
                    if i == 0:
                        input_text = [CLS_TOKEN] + sent + [SEP_TOKEN]
                    elif i != 0:
                        input_text = [SEP_TOKEN] + sent + [SEP_TOKEN]
                    wids = tokenizer.convert_tokens_to_ids(input_text)
                    article_wids.extend(wids)
                    for wid in wids:
                        if wid == dict.unk_index:
                            num_unk_token += 1
                        num_token += 1
                num_seq += 1
                article_wids = article_wids[:truncated_number] if len(article_wids) > truncated_number else article_wids
                if article_wids[-1] != dict.sep_index:
                    article_wids[-1] = dict.sep_index
                tensor = torch.IntTensor(article_wids)
                # print( dict.string_complete(tensor) )
                ds.add_item(tensor)
            ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
        elif lang == 'summary':
            for line in open(input_file, encoding='utf8'):
                article_wids = []
                max_tgt_ntokens = 500
                min_tgt_ntokens = 5
                sents = line.strip().split('<S_SEP>')
                sents = [tokenizer.tokenize(sent) for sent in sents]
                for i, sent in enumerate(sents):
                    # sometimes, there are too many token in one single sentence
                    # to be specific, there are 8 sentences in the training article longer than 512, so truncate them to 500
                    # MAX_SENT_LEN = 500
                    # if len(sent) > MAX_SENT_LEN:
                    #     sent = sent[:MAX_SENT_LEN]
                    if i != 0:
                        input_text = [SEP_TOKEN] + sent
                    else:
                        input_text = sent
                    wids = tokenizer.convert_tokens_to_ids(input_text)
                    # wtoks = tokenizer.convert_ids_to_tokens(wids)
                    # wstring = tokenizer.convert_tokens_to_string(wtoks)

                    # wids_vocab = [dict.index(word) for word in input_text]
                    # assert wids == wids_vocab, 'word indices should be the same!'
                    article_wids.extend(wids)
                    for wid in wids:
                        if wid == dict.unk_index:
                            num_unk_token += 1
                        num_token += 1

                num_seq += 1
                article_wids = article_wids[:truncated_number] if len(article_wids) > truncated_number else article_wids
                if article_wids[-1] == dict.sep_index:
                    article_wids = article_wids[:len(article_wids)-1]
                # print(article_wids)
                if len(article_wids) > truncated_number:
                    print('lang: %s, token len: %d, truncated len: %d' % (lang, len(article_wids), truncated_number))

                tensor = torch.IntTensor(article_wids)
                # print( dict.string_complete(tensor) )
                ds.add_item(tensor)
            ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, num_seq, num_token,
            100 * num_unk_token / num_token, dict.unk_word if hasattr(dict, 'unk_word') else '<no_unk_word>'))


        #
        #     n_seq_tok = [0, 0]
        #     replaced = Counter()
        #
        #     def merge_result(worker_result):
        #         replaced.update(worker_result["replaced"])
        #         n_seq_tok[0] += worker_result["nseq"]
        #         n_seq_tok[1] += worker_result["ntok"]
        #
        #     input_file = "{}{}".format(
        #         input_prefix, ("." + lang) if lang is not None else ""
        #     )
        #     offsets = Binarizer.find_offsets(input_file, num_workers)
        #     pool = None
        #     if num_workers > 1:
        #         pool = Pool(processes=num_workers - 1)
        #         for worker_id in range(1, num_workers):
        #             prefix = "{}{}".format(output_prefix, worker_id)
        #             pool.apply_async(
        #                 binarize,
        #                 (
        #                     args,
        #                     input_file,
        #                     vocab,
        #                     prefix,
        #                     lang,
        #                     offsets[worker_id],
        #                     offsets[worker_id + 1]
        #                 ),
        #                 callback=merge_result
        #             )
        #         pool.close()
        #
        #     ds = indexed_dataset.IndexedDatasetBuilder(
        #         dataset_dest_file(args, output_prefix, lang, "bin")
        #     )
        #     merge_result(
        #         Binarizer.binarize(
        #             input_file, vocab, lambda t: ds.add_item(t),
        #             offset=0, end=offsets[1]
        #         )
        #     )
        #     if num_workers > 1:
        #         pool.join()
        #         for worker_id in range(1, num_workers):
        #             prefix = "{}{}".format(output_prefix, worker_id)
        #             temp_file_path = dataset_dest_prefix(args, prefix, lang)
        #             ds.merge_file_(temp_file_path)
        #             os.remove(indexed_dataset.data_file_path(temp_file_path))
        #             os.remove(indexed_dataset.index_file_path(temp_file_path))
        #
        #     ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
        #
        #     print(
        #         "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
        #             lang,
        #             input_file,
        #             n_seq_tok[0],
        #             n_seq_tok[1],
        #             100 * sum(replaced.values()) / n_seq_tok[1],
        #             vocab.unk_word,
        #         )
        #     )

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
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    make_all(args.source_lang, src_dict)
    if target:
        make_all(args.target_lang, tgt_dict)

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
