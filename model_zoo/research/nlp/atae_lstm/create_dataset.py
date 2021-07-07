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
"""Change cor Dataset to mindrecord"""
import os
import argparse
import xml.etree.cElementTree as ET
import numpy as np
import nltk
from mindspore.mindrecord import FileWriter


class XmlParser():
    """
    parse xml dataset to cor file
    """
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.xml_sentence_category = []
        self.tolken_sentence_category = []

    def parse(self):
        """
        parse xml file to cor file
        """
        print("xml parsing...")
        self.__parse_xml_sentence_category()
        self.__parse_tolken_category()
        self.__out_tolken_sentence()
        print("parse xml success")

    def __parse_xml_sentence_category(self):
        """
        parse xml dataset to xml_sentence_category
        """
        try:
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            for sentence in root.findall('.//aspectCategories/..'):
                text = sentence.find('text').text
                aspectCategories = sentence.find('aspectCategories')
                for aspectCategory in aspectCategories.findall('aspectCategory'):
                    category = aspectCategory.get('category')
                    polarity = aspectCategory.get('polarity')
                    self.xml_sentence_category.append((text, category, polarity))
        except LookupError:
            print("no categories")

    def __parse_tolken_category(self):
        """
        parse xml_sentence to tolken_sentence_category
        """
        for i, _ in enumerate(self.xml_sentence_category):
            if len(self.xml_sentence_category[i]) == 3 and \
                    (self.xml_sentence_category[i][2] == "positive" or
                     self.xml_sentence_category[i][2] == "neutral" or
                     self.xml_sentence_category[i][2] == "negative"):
                #sentence
                tokens = nltk.word_tokenize(self.xml_sentence_category[i][0])
                joint_tokens = ""
                for j, _ in enumerate(tokens):
                    joint_tokens = joint_tokens + tokens[j] + " "
                self.tolken_sentence_category.append(joint_tokens)
                #aspect
                self.tolken_sentence_category.append(self.xml_sentence_category[i][1])
                #polarity
                if self.xml_sentence_category[i][2] == "positive":
                    self.tolken_sentence_category.append("1")
                elif self.xml_sentence_category[i][2] == "neutral":
                    self.tolken_sentence_category.append("0")
                elif self.xml_sentence_category[i][2] == "negative":
                    self.tolken_sentence_category.append("-1")
                else:
                    pass

    def __out_tolken_sentence(self):
        """
        print tolken_sentence to cor file
        """
        # category
        if self.tolken_sentence_category:
            out_path_category = self.xml_path.replace('.xml', '.cor')
            file_out_category = open(out_path_category, "w+")
            for s in self.tolken_sentence_category:
                file_out_category.write(s.lower() + '\n')
            file_out_category.close()


class Sentence():
    """docstring for sentence"""
    def __init__(self, content, target, rating, grained):
        self.content, self.target = content.lower(), target
        self.solution = np.zeros(grained, dtype=np.float32)
        self.senlength = len(self.content.split(' '))
        try:
            self.solution[int(rating)+1] = 1
        except SystemExit:
            exit()

    def stat(self, target_dict, wordlist, grained=3):
        """statistical"""
        data, data_target, i = [], [], 0
        solution = np.zeros((self.senlength, grained), dtype=np.float32)
        for word in self.content.split(' '):
            data.append(wordlist[word])
            try:
                pol = Lexicons_dict[word]
                solution[i][pol + 1] = 1
            except NameError:
                pass
            i = i + 1
        for word in self.target.split(' '):
            data_target.append(wordlist[word])
        return {'seqs': data,
                'target': data_target,
                'solution': np.array([self.solution]),
                'target_index': self.get_target(target_dict)}

    def get_target(self, dict_target):
        """
        target
        """
        return dict_target[self.target]


class DataManager():
    """create mindrecord dataset"""
    def __init__(self, dataset, grained=3):
        self.fileList = ['train', 'test', 'dev']
        self.origin = {}
        self.wordlist = {}
        self.data = {}
        for fname in self.fileList:
            data = []
            with open('%s/%s.cor' % (dataset, fname)) as f:
                sentences = f.readlines()
                for i in range(int(len(sentences)/3)):
                    content, target = sentences[i * 3].strip(), sentences[i * 3 + 1].strip()
                    rating = sentences[i * 3 + 2].strip()
                    sentence = Sentence(content, target, rating, grained)
                    data.append(sentence)
            self.origin[fname] = data
        self.gen_target()

    def gen_word(self):
        """Statistical characters"""
        wordcount = {}
        def sta(sentence):
            """
            Sentence Statistical
            """
            for word in sentence.content.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except LookupError:
                    wordcount[word] = 1
            for word in sentence.target.split(' '):
                try:
                    wordcount[word] = wordcount.get(word, 0) + 1
                except LookupError:
                    wordcount[word] = 1

        for fname in self.fileList:
            for sent in self.origin[fname]:
                sta(sent)
        words = wordcount.items()
        sorted(words, key=lambda x: x[1], reverse=True)
        self.wordlist = {item[0]: index + 1 for index, item in enumerate(words)}
        return self.wordlist

    def gen_target(self, threshold=5):
        """Statistical aspect"""
        self.dict_target = {}
        for fname in self.fileList:
            for sent in self.origin[fname]:
                if sent.target in self.dict_target:
                    self.dict_target[sent.target] = self.dict_target[sent.target] + 1
                else:
                    self.dict_target[sent.target] = 1
        i = 0
        for (key, val) in self.dict_target.items():
            if val < threshold:
                self.dict_target[key] = 0
            else:
                self.dict_target[key] = i
                i = i + 1
        return self.dict_target

    def gen_data(self, grained=3):
        """all data"""
        if grained != 3:
            print("only support 3")

        for fname in self.fileList:
            self.data[fname] = []
            for sent in self.origin[fname]:
                self.data[fname].append(sent.stat(self.dict_target, self.wordlist))
        return self.data['train'], self.data['dev'], self.data['test']

    def word2vec_pre_select(self, mdict, word2vec_file_path, save_vec_file_path):
        """word to vector"""
        list_seledted = ['']
        line = ''
        with open(word2vec_file_path) as f:
            for line in f:
                tmp = line.strip().split(' ', 1)
                if tmp[0] in mdict:
                    list_seledted.append(line.strip())
        list_seledted[0] = str(len(list_seledted)-1) + ' ' + str(len(line.strip().split())-1)
        open(save_vec_file_path, 'w').write('\n'.join(list_seledted))


def _convert_to_mindrecord(mindrecord_path, data):
    """
    convert cor dataset to mindrecord dataset
    """
    print("convert to mindrecord...")

    content = []
    sen_len = []
    aspect = []
    solution = []
    aspect_index = []

    for info in data:
        content.append(info['seqs'])
        aspect.append(info['target'])
        sen_len.append([len(info['seqs'])])
        solution.append(info['solution'])
        aspect_index.append(info['target_index'])

    padded_content = np.zeros([len(content), 50])
    for index, seq in enumerate(content):
        if len(seq) <= 50:
            padded_content[index, 0:len(seq)] = seq
        else:
            padded_content[index] = seq[0:50]

    content = padded_content

    if os.path.exists(mindrecord_path):
        os.remove(mindrecord_path)
        os.remove(mindrecord_path + ".db")

    # schema
    schema_json = {"content": {"type": "int32", "shape": [-1]},
                   "sen_len": {"type": "int32"},
                   "aspect": {"type": "int32"},
                   "solution": {"type": "int32", "shape": [-1]}}

    data_list = []
    for i, _ in enumerate(content):
        sample = {"content": content[i],
                  "sen_len": int(sen_len[i][0]),
                  "aspect": int(aspect[i][0]),
                  "solution": solution[i][0]}
        data_list.append(sample)

    writer = FileWriter(mindrecord_path, shard_num=1)
    writer.add_schema(schema_json, "lstm_schema")
    writer.write_raw_data(data_list)
    writer.commit()


def wordlist_to_glove_weight(wordlist, glove_file):
    """load glove word vector"""
    glove_word_dict = {}
    with open(glove_file) as f:
        line = f.readline()
        while line:
            array = line.split(' ')
            word = array[0]
            glove_word_dict[word] = array[1:301]
            line = f.readline()

    weight = np.zeros((len(wordlist)+1, 300)).astype(np.float32)+0.01
    unfound_count = 0
    for word, i in wordlist.items():
        word = word.strip()
        if word in glove_word_dict:
            weight[i] = glove_word_dict[word]
        else:
            unfound_count += 1

    print("not found in glove: ", unfound_count)
    print(np.shape(weight))
    print(weight.dtype)

    np.savez('./data/weight.npz', weight=weight)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create AttentionLSTM dataset.')
    parser.add_argument("--data_folder", type=str, required=True, help="data folder.")
    parser.add_argument("--glove_file", type=str, required=True, help="glove 300d file.")
    parser.add_argument("--train_data", type=str, required=True, help="train data.")
    parser.add_argument("--eval_data", type=str, required=True, help="eval data.")
    args, _ = parser.parse_known_args()

    # train
    train_data_path = os.path.join(args.data_folder, 'train.cor')
    if train_data_path.endswith('.xml'):
        xml_parser = XmlParser(train_data_path)
        xml_parser.parse()
    # test
    test_data_path = os.path.join(args.data_folder, 'test.cor')
    if test_data_path.endswith('.xml'):
        xml_parser = XmlParser(test_data_path)
        xml_parser.parse()
    # dev
    dev_data_path = os.path.join(args.data_folder, 'dev.cor')
    if dev_data_path.endswith('.xml'):
        xml_parser = XmlParser(dev_data_path)
        xml_parser.parse()

    data_all = DataManager(args.data_folder)
    word_list = data_all.gen_word()
    print("word_list: ", type(word_list))

    wordlist_to_glove_weight(word_list, args.glove_file)

    train_data, dev_data, test_data = data_all.gen_data(grained=3)

    _convert_to_mindrecord(args.train_data, train_data)
    _convert_to_mindrecord(args.eval_data, test_data)
