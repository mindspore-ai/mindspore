# Copyright 2020 Huawei Technologies Co., Ltd
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
"""FastText data preprocess"""
import csv
import os
import re
import argparse
import pprint
import ast
import html
import numpy as np
import spacy
from sklearn.feature_extraction import FeatureHasher
from mindspore.mindrecord import FileWriter


class FastTextDataPreProcess():
    """FastText data preprocess"""

    def __init__(self, train_path,
                 test_file,
                 max_length,
                 class_num,
                 ngram,
                 train_feature_dict,
                 buckets,
                 test_feature_dict,
                 test_bucket,
                 is_hashed,
                 feature_size):
        self.train_path = train_path
        self.test_path = test_file
        self.max_length = max_length
        self.class_num = class_num
        self.train_feature_dict = train_feature_dict
        self.test_feature_dict = test_feature_dict
        self.test_bucket = test_bucket
        self.is_hashed = is_hashed
        self.feature_size = feature_size
        self.buckets = buckets
        self.ngram = ngram
        self.text_greater = '>'
        self.text_less = '<'
        self.word2vec = dict()
        self.vec2words = dict()
        self.non_str = '\\'
        self.end_string = ['.', '?', '!']
        self.word2vec['PAD'] = 0
        self.vec2words[0] = 'PAD'
        self.word2vec['UNK'] = 1
        self.vec2words[1] = 'UNK'
        self.str_html = re.compile(r'<[^>]+>')

    def load(self):
        """data preprocess loader"""
        train_dataset_list = []
        test_dataset_list = []
        spacy_nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
        spacy_nlp.add_pipe(spacy_nlp.create_pipe('sentencizer'))

        with open(self.train_path, 'r', newline='', encoding='utf-8') as src_file:
            reader = csv.reader(src_file, delimiter=",", quotechar='"')
            for _, _pair_sen in enumerate(reader):
                label_idx = int(_pair_sen[0]) - 1
                if len(_pair_sen) == 3:
                    src_tokens = self.input_preprocess(src_text1=_pair_sen[1],
                                                       src_text2=_pair_sen[2],
                                                       spacy_nlp=spacy_nlp,
                                                       train_mode=True)
                    src_tokens_length = len(src_tokens)
                elif len(_pair_sen) == 2:
                    src_tokens = self.input_preprocess(src_text1=_pair_sen[1],
                                                       src_text2=None,
                                                       spacy_nlp=spacy_nlp,
                                                       train_mode=True)
                    src_tokens_length = len(src_tokens)
                elif len(_pair_sen) == 4:
                    if _pair_sen[2]:
                        sen_o_t = _pair_sen[1] + ' ' + _pair_sen[2]
                    else:
                        sen_o_t = _pair_sen[1]
                    src_tokens = self.input_preprocess(src_text1=sen_o_t,
                                                       src_text2=_pair_sen[3],
                                                       spacy_nlp=spacy_nlp,
                                                       train_mode=True)
                    src_tokens_length = len(src_tokens)

                train_dataset_list.append([src_tokens, src_tokens_length, label_idx])

        print("Begin to process test data...")
        with open(self.test_path, 'r', newline='', encoding='utf-8') as test_file:
            reader2 = csv.reader(test_file, delimiter=",", quotechar='"')
            for _, _test_sen in enumerate(reader2):
                label_idx = int(_test_sen[0]) - 1
                if len(_test_sen) == 3:
                    src_tokens = self.input_preprocess(src_text1=_test_sen[1],
                                                       src_text2=_test_sen[2],
                                                       spacy_nlp=spacy_nlp,
                                                       train_mode=False)
                    src_tokens_length = len(src_tokens)
                elif len(_test_sen) == 2:
                    src_tokens = self.input_preprocess(src_text1=_test_sen[1],
                                                       src_text2=None,
                                                       spacy_nlp=spacy_nlp,
                                                       train_mode=False)
                    src_tokens_length = len(src_tokens)
                elif len(_test_sen) == 4:
                    if _test_sen[2]:
                        sen_o_t = _test_sen[1] + ' ' + _test_sen[2]
                    else:
                        sen_o_t = _test_sen[1]
                    src_tokens = self.input_preprocess(src_text1=sen_o_t,
                                                       src_text2=_test_sen[3],
                                                       spacy_nlp=spacy_nlp,
                                                       train_mode=False)
                    src_tokens_length = len(src_tokens)

                test_dataset_list.append([src_tokens, src_tokens_length, label_idx])

        if self.is_hashed:
            print("Begin to Hashing Trick......")
            features_num = self.feature_size
            fh = FeatureHasher(n_features=features_num, alternate_sign=False)
            print("FeatureHasher features..", features_num)
            self.hash_trick(fh, train_dataset_list)
            self.hash_trick(fh, test_dataset_list)
            print("Hashing Done....")

        # pad train dataset
        train_dataset_list_length = len(train_dataset_list)
        test_dataset_list_length = len(test_dataset_list)
        for l in range(train_dataset_list_length):
            bucket_length = self._get_bucket_length(train_dataset_list[l][0], self.buckets)
            while len(train_dataset_list[l][0]) < bucket_length:
                train_dataset_list[l][0].append(self.word2vec['PAD'])
            train_dataset_list[l][1] = len(train_dataset_list[l][0])
        # pad test dataset
        for j in range(test_dataset_list_length):
            test_bucket_length = self._get_bucket_length(test_dataset_list[j][0], self.test_bucket)
            while len(test_dataset_list[j][0]) < test_bucket_length:
                test_dataset_list[j][0].append(self.word2vec['PAD'])
            test_dataset_list[j][1] = len(test_dataset_list[j][0])

        train_example_data = []
        test_example_data = []
        for idx in range(train_dataset_list_length):
            train_example_data.append({
                "src_tokens": train_dataset_list[idx][0],
                "src_tokens_length": train_dataset_list[idx][1],
                "label_idx": train_dataset_list[idx][2],
            })
            for key in self.train_feature_dict:
                if key == train_example_data[idx]['src_tokens_length']:
                    self.train_feature_dict[key].append(train_example_data[idx])
        for h in range(test_dataset_list_length):
            test_example_data.append({
                "src_tokens": test_dataset_list[h][0],
                "src_tokens_length": test_dataset_list[h][1],
                "label_idx": test_dataset_list[h][2],
            })
            for key in self.test_feature_dict:
                if key == test_example_data[h]['src_tokens_length']:
                    self.test_feature_dict[key].append(test_example_data[h])
        print("train vocab size is ", len(self.word2vec))

        return self.train_feature_dict, self.test_feature_dict

    def input_preprocess(self, src_text1, src_text2, spacy_nlp, train_mode):
        """data preprocess func"""
        src_text1 = src_text1.strip()
        if src_text1 and src_text1[-1] not in self.end_string:
            src_text1 = src_text1 + '.'

        if src_text2:
            src_text2 = src_text2.strip()
            sent_describe = src_text1 + ' ' + src_text2
        else:
            sent_describe = src_text1
        if self.non_str in sent_describe:
            sent_describe = sent_describe.replace(self.non_str, ' ')

        sent_describe = html.unescape(sent_describe)

        if self.text_less in sent_describe and self.text_greater in sent_describe:
            sent_describe = self.str_html.sub('', sent_describe)

        doc = spacy_nlp(sent_describe)
        bows_token = [token.text for token in doc]

        try:
            tagged_sent_desc = '<p> ' + ' </s> '.join([s.text for s in doc.sents]) + ' </p>'
        except ValueError:
            tagged_sent_desc = '<p> ' + sent_describe + ' </p>'
        doc = spacy_nlp(tagged_sent_desc)
        ngrams = self.generate_gram([token.text for token in doc], num=self.ngram)

        bo_ngrams = bows_token + ngrams

        if train_mode is True:
            for ngms in bo_ngrams:
                idx = self.word2vec.get(ngms)
                if idx is None:
                    idx = len(self.word2vec)
                    self.word2vec[ngms] = idx
                    self.vec2words[idx] = ngms

        processed_out = [self.word2vec[ng] if ng in self.word2vec else self.word2vec['UNK'] for ng in bo_ngrams]

        return processed_out

    def _get_bucket_length(self, x, bts):
        x_len = len(x)
        for index in range(1, len(bts)):
            if bts[index - 1] < x_len <= bts[index]:
                return bts[index]
        return bts[0]

    def generate_gram(self, words, num=2):

        return [' '.join(words[i: i + num]) for i in range(len(words) - num + 1)]

    def count2dict(self, lst):
        count_dict = dict()
        for m in lst:
            if str(m) in count_dict:
                count_dict[str(m)] += 1
            else:
                count_dict[str(m)] = 1
        return count_dict

    def hash_trick(self, hashing, input_data):
        trans = hashing.transform((self.count2dict(e[0]) for e in input_data))
        for htr, e in zip(trans, input_data):
            sparse2bow = list()
            for idc, d in zip(htr.indices, htr.data):
                for _ in range(int(d)):
                    sparse2bow.append(idc + 1)
            e[0] = sparse2bow


def write_to_mindrecord(data, path, shared_num=1):
    """generate mindrecord"""
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    writer = FileWriter(path, shared_num)
    data_schema = {
        "src_tokens": {"type": "int32", "shape": [-1]},
        "src_tokens_length": {"type": "int32", "shape": [-1]},
        "label_idx": {"type": "int32", "shape": [-1]}
    }
    writer.add_schema(data_schema, "fasttext")
    for item in data:
        item['src_tokens'] = np.array(item['src_tokens'], dtype=np.int32)
        item['src_tokens_length'] = np.array(item['src_tokens_length'], dtype=np.int32)
        item['label_idx'] = np.array(item['label_idx'], dtype=np.int32)
        writer.write_raw_data([item])
    writer.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True, help='train dataset file path')
    parser.add_argument('--test_file', type=str, required=True, help='test dataset file path')
    parser.add_argument('--class_num', type=int, required=True, help='Dataset classe number')
    parser.add_argument('--ngram', type=int, default=2, required=False)
    parser.add_argument('--max_len', type=int, required=False, help='max length sentence in dataset')
    parser.add_argument('--bucket', type=ast.literal_eval, default=[64, 128, 467], help='bucket sequence length.')
    parser.add_argument('--test_bucket', type=ast.literal_eval, default=[467], help='bucket sequence length.')
    parser.add_argument('--is_hashed', type=bool, default=False, help='add hash trick for dataset')
    parser.add_argument('--feature_size', type=int, default=10000000, help='hash feature size')

    args = parser.parse_args()
    pprint.PrettyPrinter().pprint(args.__dict__)
    train_feature_dicts = {}
    for i in args.bucket:
        train_feature_dicts[i] = []
    test_feature_dicts = {}
    for i in args.test_bucket:
        test_feature_dicts[i] = []

    g_d = FastTextDataPreProcess(train_path=args.train_file,
                                 test_file=args.test_file,
                                 max_length=args.max_len,
                                 ngram=args.ngram,
                                 class_num=args.class_num,
                                 train_feature_dict=train_feature_dicts,
                                 buckets=args.bucket,
                                 test_feature_dict=test_feature_dicts,
                                 test_bucket=args.test_bucket,
                                 is_hashed=args.is_hashed,
                                 feature_size=args.feature_size)
    train_data_example, test_data_example = g_d.load()
    print("Data preprocess done")
    print("Writing train data to MindRecord file......")

    for i in args.bucket:
        write_to_mindrecord(train_data_example[i], './train_dataset_bs_' + str(i) + '.mindrecord', 1)

    print("Writing test data to MindRecord file.....")
    for k in args.test_bucket:
        write_to_mindrecord(test_data_example[k], './test_dataset_bs_' + str(k) + '.mindrecord', 1)

    print("All done.....")
