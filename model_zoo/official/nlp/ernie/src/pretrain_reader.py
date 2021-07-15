# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""""
Ernie pretrain data reader
"""

import collections
import re
import argparse
import random
import jieba
import numpy as np
from opencc import OpenCC
from mindspore.mindrecord import FileWriter
from src.tokenizer import convert_to_unicode, CharTokenizer
from src.utils import get_file_list

class ErnieDataReader:
    """Ernie data reader"""
    def __init__(self,
                 file_list,
                 vocab_path,
                 short_seq_prob,
                 masked_word_prob,
                 max_predictions_per_seq,
                 dupe_factor,
                 max_seq_len=512,
                 random_seed=1,
                 do_lower_case=True,
                 generate_neg_sample=False):
        # short_seq_prob, masked_word_prob, max_predictions_per_seq, vocab_words

        self.vocab = self.load_vocab(vocab_path)
        self.tokenizer = CharTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)

        self.short_seq_prob = short_seq_prob
        self.masked_word_prob = masked_word_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor

        self.file_list = file_list

        self.max_seq_len = max_seq_len
        self.generate_neg_sample = generate_neg_sample

        self.global_rng = random.Random(random_seed)

        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        fin = open(vocab_file)
        for num, line in enumerate(fin):
            items = convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        return vocab

    def random_pair_neg_samples(self, pos_samples):
        """ randomly generate negative samples using pos_samples
            Args:
                pos_samples: list of positive samples

            Returns:
                neg_samples: list of negative samples
        """
        np.random.shuffle(pos_samples)
        num_sample = len(pos_samples)
        neg_samples = []
        miss_num = 0

        def split_sent(sample, max_len, sep_id):
            token_ids, _, _, seg_labels, _ = sample
            sep_index = token_ids.index(sep_id)
            left_len = sep_index - 1
            if left_len <= max_len:
                return (token_ids[1:sep_index], seg_labels[1:sep_index])
            return [
                token_ids[sep_index + 1:-1], seg_labels[sep_index + 1:-1]
            ]

        for i in range(num_sample):
            pair_index = (i + 1) % num_sample
            left_tokens, left_seg_labels = split_sent(
                pos_samples[i], (self.max_seq_len - 3) // 2, self.sep_id)
            right_tokens, right_seg_labels = split_sent(
                pos_samples[pair_index],
                self.max_seq_len - 3 - len(left_tokens), self.sep_id)

            token_seq = [self.cls_id] + left_tokens + [self.sep_id] + \
                    right_tokens + [self.sep_id]
            if len(token_seq) > self.max_seq_len:
                miss_num += 1
                continue
            type_seq = [0] * (len(left_tokens) + 2) + [1] * (len(right_tokens) +
                                                             1)
            pos_seq = range(len(token_seq))
            seg_label_seq = [-1] + left_seg_labels + [-1] + right_seg_labels + [
                -1
            ]

            assert len(token_seq) == len(type_seq) == len(pos_seq) == len(seg_label_seq), \
                    "[ERROR]len(src_id) == lne(sent_id) == len(pos_id) must be True"
            neg_samples.append([token_seq, type_seq, pos_seq, seg_label_seq, 0])

        return neg_samples, miss_num

    def mixin_negative_samples(self, sample_generator, buffer=1000):
        """ 1. generate negative samples by randomly group sentence_1 and sentence_2 of positive samples
            2. combine negative samples and positive samples

            Args:
                pos_sample_generator: a generator producing a parsed positive sample,
                which is a list: [token_ids, sent_ids, pos_ids, 1]
            Returns:
                sample: one sample from shuffled positive samples and negative samples
        """
        pos_samples = []
        neg_samples = []
        num_total_miss = 0
        pos_sample_num = 0
        try:
            while True:
                while len(pos_samples) < buffer:
                    sample = next(sample_generator)
                    next_sentence_label = sample[-1]
                    if next_sentence_label == 1:
                        pos_samples.append(sample)
                        pos_sample_num += 1
                    else:
                        neg_samples.append(sample)

                new_neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples + new_neg_samples
                pos_samples = []
                neg_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
        except StopIteration:
            print("stopiteration: reach end of file")
            if len(pos_samples) == 1:
                yield pos_samples[0]
            elif not pos_samples:
                yield None
            else:
                new_neg_samples, miss_num = self.random_pair_neg_samples(
                    pos_samples)
                num_total_miss += miss_num
                samples = pos_samples + neg_samples + new_neg_samples
                pos_samples = []
                neg_samples = []
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
            print("miss_num:%d\tideal_total_sample_num:%d\tmiss_rate:%f" %
                  (num_total_miss, pos_sample_num * 2,
                   num_total_miss / (pos_sample_num * 2)))

    def shuffle_samples(self, sample_generator, buffer=1000):
        """shuffle samples"""
        samples = []
        try:
            while True:
                while len(samples) < buffer:
                    sample = next(sample_generator)
                    samples.append(sample)
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample
                samples = []
        except StopIteration:
            print("stopiteration: reach end of file")
            if not samples:
                yield None
            else:
                np.random.shuffle(samples)
                for sample in samples:
                    yield sample

    def data_generator(self):
        """
        data_generator
        """
        mask_word = (np.random.random() < float(self.masked_word_prob))
        if mask_word:
            self.mask_type = "mask_word"
        else:
            self.mask_type = "mask_char"

        sample_generator = self.create_training_instances()
        if self.generate_neg_sample:
            sample_generator = self.mixin_negative_samples(
                sample_generator)
        else:
            #shuffle buffered sample
            sample_generator = self.shuffle_samples(
                sample_generator)

        for sample in sample_generator:
            if sample is None:
                continue
            sample.append(mask_word)
            yield sample

    def create_training_instances(self):
        """Create `TrainingInstance`s from raw text."""
        p1 = re.compile('<doc (.*)>')
        p2 = re.compile('</doc>')
        cc = OpenCC('t2s')
        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        for input_file in self.file_list:
            all_documents = [[]]
            all_documents_segs = [[]]
            with open(input_file, "r", errors='ignore') as reader:
                while True:
                    line = reader.readline()
                    if not line:
                        break
                    if p2.match(line):
                        all_documents.append([])
                        all_documents_segs.append([])
                    line = p1.sub('', line)
                    line = p2.sub('', line)
                    line = cc.convert(line)
                    line = convert_to_unicode(line)
                    line = line.strip()

                    segs = self.get_word_segs(line)
                    tokens = self.tokenizer.tokenize(segs)
                    seg_labels = [0 if "##" not in x else 1 for x in tokens]

                    if tokens:
                        all_documents[-1].append(tokens)
                        all_documents_segs[-1].append(seg_labels)
            # Remove empty documents
            all_documents = [x for x in all_documents if x]
            all_documents_segs = [x for x in all_documents_segs if x]

            instances = []
            for _ in range(self.dupe_factor):
                for document_index in range(len(all_documents)):
                    instances.extend(
                        self.create_instances_from_document(
                            all_documents, all_documents_segs, document_index))

            self.global_rng.shuffle(instances)
            for instance in instances:
                yield instance

    def create_instances_from_document(self, all_documents, all_documents_segs, document_index):
        """Creates `TrainingInstance`s for a single document."""
        def get_random_document_index(all_documents, document_index):
            for _ in range(10):
                random_document_index = self.global_rng.randint(0, len(all_documents) - 1)
                if random_document_index != document_index:
                    break
            return random_document_index

        def get_tokens_b(random_document, random_document_segs, tokens_b,
                         tokens_b_segs, random_start, target_b_length):
            for j in range(random_start, len(random_document)):
                tokens_b.extend(random_document[j])
                tokens_b_segs.extend(random_document_segs[j])
                if len(tokens_b) >= target_b_length:
                    break
            return tokens_b, tokens_b_segs

        document = all_documents[document_index]
        document_segs = all_documents_segs[document_index]
        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = self.max_seq_len - 3

        # We *usually* want to fill up the entire sequence since we are padding
        # to `max_seq_length` anyways, so short sequences are generally wasted
        # computation. However, we *sometimes*
        # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and fine-tuning.
        # The `target_seq_length` is just a rough target however, whereas
        # `max_seq_length` is a hard limit.
        target_seq_length = max_num_tokens
        if self.global_rng.random() < self.short_seq_prob:
            target_seq_length = self.global_rng.randint(2, max_num_tokens)

        # We DON'T just concatenate all of the tokens from a document into a long
        # sequence and choose an arbitrary split point because this would make the
        # next sentence prediction task too easy. Instead, we split the input into
        # segments "A" and "B" based on the actual "sentences" provided by the user
        # input.
        instances = []
        current_chunk = []
        current_chunk_segs = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_chunk_segs.append(document_segs[i])
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = self.global_rng.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    tokens_a_segs = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                        tokens_a_segs.extend(current_chunk_segs[j])

                    tokens_b = []
                    tokens_b_segs = []
                    # Random next
                    is_random_next = False
                    if len(current_chunk) == 1 or self.global_rng.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        random_document_index = get_random_document_index(all_documents, document_index)

                        random_document = all_documents[random_document_index]
                        random_document_segs = all_documents_segs[random_document_index]
                        random_start = self.global_rng.randint(0, len(random_document) - 1)
                        tokens_b, tokens_b_segs = get_tokens_b(random_document, random_document_segs, tokens_b,
                                                               tokens_b_segs, random_start, target_b_length)
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                            tokens_b_segs.extend(current_chunk_segs[j])

                    self.truncate_seq_pair(tokens_a, tokens_b, tokens_a_segs, tokens_b_segs,
                                           max_num_tokens, self.global_rng)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    tokens = []
                    seg_labels = []
                    segment_ids = []

                    tokens.append("[CLS]")
                    seg_labels.append(-1)
                    segment_ids.append(0)
                    for idx, token in enumerate(tokens_a):
                        tokens.append(token)
                        segment_ids.append(0)
                        seg_labels.append(tokens_a_segs[idx])

                    tokens.append("[SEP]")
                    segment_ids.append(0)
                    seg_labels.append(-1)

                    for idx, token in enumerate(tokens_b):
                        tokens.append(token)
                        segment_ids.append(1)
                        seg_labels.append(tokens_b_segs[idx])

                    tokens.append("[SEP]")
                    segment_ids.append(1)
                    seg_labels.append(-1)

                    position_ids = [i for i in range(len(tokens))]
                    token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                    assert len(tokens) == len(segment_ids)
                    assert len(segment_ids) == len(seg_labels)
                    next_sentence_label = 0 if is_random_next else 1

                    instance = [token_ids, segment_ids, position_ids, seg_labels, next_sentence_label]
                    instances.append(instance)

                current_chunk = []
                current_chunk_segs = []
                current_length = 0
            i += 1
        return instances

    def truncate_seq_pair(self, tokens_a, tokens_b, tokens_a_segs, tokens_b_segs, max_num_tokens, rng):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            trunc_tokens_segs = tokens_a_segs if len(tokens_a) > len(tokens_b) else tokens_b_segs
            assert len(trunc_tokens) >= 1
            assert len(trunc_tokens_segs) >= 1
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
                del trunc_tokens_segs[0]
            else:
                trunc_tokens.pop()
                trunc_tokens_segs.pop()

    def mask(self,
             token_ids,
             seg_labels,
             mask_word_tag,
             vocab_size,
             CLS=1,
             SEP=2,
             MASK=3):
        """
        Add mask for batch_tokens, return out, mask_label, mask_pos;
        Note: mask_pos responding the batch_tokens after padded;
        """
        num_to_predict = min(self.max_predictions_per_seq,
                             max(1, int(round(len(token_ids) * self.masked_word_prob))))
        mask_label = []
        mask_pos = []
        prob_mask = np.random.rand(len(token_ids))
        # Note: the first token is [CLS], so [low=1]
        replace_ids = np.random.randint(1, high=vocab_size, size=len(token_ids))

        if mask_word_tag:
            beg = 0
            for token_index, token in enumerate(token_ids):
                seg_label = seg_labels[token_index]
                if seg_label == 1:
                    continue
                if beg == 0:
                    if seg_label != -1:
                        beg = token_index
                    continue

                prob = prob_mask[beg]
                if prob > 0.15:
                    pass
                else:
                    for index in range(beg, token_index):
                        prob = prob_mask[index]
                        base_prob = 1.0
                        if index == beg:
                            base_prob = 0.15
                        if base_prob * 0.2 < prob <= base_prob:
                            mask_label.append(token_ids[index])
                            token_ids[index] = MASK
                            mask_pos.append(index)
                        elif base_prob * 0.1 < prob <= base_prob * 0.2:
                            mask_label.append(token_ids[index])
                            token_ids[index] = replace_ids[index]
                            mask_pos.append(index)
                        else:
                            mask_label.append(token_ids[index])
                            mask_pos.append(index)

                        if len(mask_label) >= num_to_predict:
                            return token_ids, mask_label, mask_pos
                if seg_label == -1:
                    beg = 0
                else:
                    beg = token_index
        else:
            for token_index, token in enumerate(token_ids):
                prob = prob_mask[token_index]
                if prob > 0.15:
                    continue
                elif 0.03 < prob <= 0.15:
                    # mask
                    if token not in (SEP, CLS):
                        mask_label.append(token_ids[token_index])
                        token_ids[token_index] = MASK
                        mask_pos.append(token_index)
                elif 0.015 < prob <= 0.03:
                    # random replace
                    if token not in (SEP, CLS):
                        mask_label.append(token_ids[token_index])
                        token_ids[token_index] = replace_ids[token_index]
                        mask_pos.append(token_index)
                else:
                    # keep the original token
                    if token not in (SEP, CLS):
                        mask_label.append(token_ids[token_index])
                        mask_pos.append(token_index)
                if len(mask_label) >= num_to_predict:
                    break

        return token_ids, mask_label, mask_pos

    def _convert_example_to_record(self, example):
        """convert example to record"""
        input_ids, segment_ids, _, seg_labels, next_sentence_label, mask_word = example
        vocab_size = len(self.vocab.keys())
        input_ids, masked_lm_ids, masked_lm_positions = self.mask(input_ids, seg_labels, mask_word, vocab_size)

        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        masked_lm_weights = [1.0] * len(masked_lm_ids)
        while len(masked_lm_positions) < self.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)
        return input_ids, input_mask, segment_ids, next_sentence_label, \
            masked_lm_positions, masked_lm_ids, masked_lm_weights

    def get_word_segs(self, sentence):
        segs = jieba.lcut(sentence)
        return " ".join(segs)

    def file_based_convert_examples_to_features(self, output_file, shard_num):
        """"Convert a set of `InputExample`s to a MindDataset file."""
        writer = FileWriter(file_name=output_file, shard_num=shard_num)
        nlp_schema = {
            "input_ids": {"type": "int64", "shape": [-1]},
            "input_mask": {"type": "int64", "shape": [-1]},
            "token_type_id": {"type": "int64", "shape": [-1]},
            "next_sentence_labels": {"type": "int64", "shape": [-1]},
            "masked_lm_positions": {"type": "int64", "shape": [-1]},
            "masked_lm_ids": {"type": "int64", "shape": [-1]},
            "masked_lm_weights": {"type": "int64", "shape": [-1]},
        }
        writer.add_schema(nlp_schema, "proprocessed pretrain dataset")
        data = []
        index = 0
        for example in self.data_generator():
            if index % 1000 == 0:
                print("Writing example %d" % index)
            record = self._convert_example_to_record(example)
            sample = {
                "input_ids": np.array(record[0], dtype=np.int64),
                "input_mask": np.array(record[1], dtype=np.int64),
                "token_type_id": np.array(record[2], dtype=np.int64),
                "next_sentence_labels": np.array([record[3]], dtype=np.int64),
                "masked_lm_positions": np.array(record[4], dtype=np.int64),
                "masked_lm_ids": np.array(record[5], dtype=np.int64),
                "masked_lm_weights": np.array(record[6], dtype=np.int64),
            }
            data.append(sample)
            index += 1
        writer.write_raw_data(data)
        writer.commit()

def main():
    parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
    parser.add_argument("--vocab_path", type=str, default="pretrain_models/converted/vocab.txt", help="vocab file")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="The maximum total input sequence length after WordPiece tokenization. "
                        "Sequences longer than this will be truncated, and sequences shorter "
                        "than this will be padded.")
    parser.add_argument("--do_lower_case", type=str, default="true",
                        help="Whether to lower case the input text. "
                        "Should be True for uncased models and False for cased models.")
    parser.add_argument("--random_seed", type=int, default=0, help="random seed number")
    parser.add_argument("--short_seq_prob", type=float, default=0.1, help="random seed number")
    parser.add_argument("--masked_word_prob", type=float, default=0.15, help="random seed number")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20, help="random seed number")
    parser.add_argument("--dupe_factor", type=int, default=10, help="random seed number")

    parser.add_argument("--generate_neg_sample", type=str, default="true", help="random seed number")

    parser.add_argument("--input_file", type=str, default="", help="raw data file")
    parser.add_argument("--output_file", type=str, default="", help="minddata file")
    parser.add_argument("--shard_num", type=int, default=1, help="output file shard number")
    args_opt = parser.parse_args()

    file_list = get_file_list(args_opt.input_file)

    reader = ErnieDataReader(file_list=file_list,
                             vocab_path=args_opt.vocab_path,
                             short_seq_prob=args_opt.short_seq_prob,
                             masked_word_prob=args_opt.masked_word_prob,
                             max_predictions_per_seq=args_opt.max_predictions_per_seq,
                             dupe_factor=args_opt.dupe_factor,
                             max_seq_len=args_opt.max_seq_len,
                             random_seed=args_opt.random_seed,
                             do_lower_case=(args_opt.do_lower_case == 'true'),
                             generate_neg_sample=(args_opt.generate_neg_sample == 'true'))

    reader.file_based_convert_examples_to_features(output_file=args_opt.output_file,
                                                   shard_num=args_opt.shard_num)

if __name__ == "__main__":
    main()
