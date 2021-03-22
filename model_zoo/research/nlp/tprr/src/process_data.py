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
"""
Process Data.

"""

import json
import pickle as pkl

from transformers import BertTokenizer

from src.utils import get_new_title, get_raw_title


class DataGen:
    """data generator"""

    def __init__(self, config):
        """init function"""
        self.wiki_path = config.wiki_path
        self.dev_path = config.dev_path
        self.dev_data_path = config.dev_data_path
        self.num_docs = config.num_docs
        self.max_q_len = config.q_len
        self.max_doc_len = config.d_len
        self.max_seq_len2 = config.s_len
        self.vocab = config.vocab_path
        self.onehop_num = config.onehop_num

        self.data_db, self.dev_data, self.q_doc_text = self.load_data()
        self.query2id, self.q_gold = self.process_data()
        self.id2title, self.id2doc, self.query_id_list, self.id2query = self.load_id2()

    def load_data(self):
        """load data"""
        print('**********************  loading data  ********************** ')
        f_wiki = open(self.wiki_path, 'rb')
        f_train = open(self.dev_path, 'rb')
        f_doc = open(self.dev_data_path, 'rb')
        data_db = pkl.load(f_wiki, encoding="gbk")
        dev_data = json.load(f_train)
        q_doc_text = pkl.load(f_doc, encoding='gbk')
        f_wiki.close()
        f_train.close()
        f_doc.close()
        return data_db, dev_data, q_doc_text

    def process_data(self):
        """process data"""
        query2id = {}
        q_gold = {}
        for onedata in self.dev_data:
            if onedata['question'] not in query2id:
                q_gold[onedata['_id']] = {}
                query2id[onedata['question']] = onedata['_id']
                gold_path = []
                for item in onedata['path']:
                    gold_path.append(get_raw_title(item))
                q_gold[onedata['_id']]['title'] = gold_path
                gold_text = []
                for item in gold_path:
                    gold_text.append(self.data_db[get_new_title(item)]['text'])
                q_gold[onedata['_id']]['text'] = gold_text
        return query2id, q_gold

    def load_id2(self):
        """load dev data"""
        with open(self.dev_data_path, 'rb') as f:
            temp_dev_dic = pkl.load(f, encoding='gbk')
        id2title = {}
        id2doc = {}
        id2query = {}
        query_id_list = []
        for q_id in temp_dev_dic:
            id2title[q_id] = temp_dev_dic[q_id]['title']
            id2doc[q_id] = temp_dev_dic[q_id]['text']
            query_id_list.append(q_id)
            id2query[q_id] = temp_dev_dic[q_id]['query']
        return id2title, id2doc, query_id_list, id2query

    def get_query2id(self, query):
        """get query id"""
        output_list = []
        for item in query:
            output_list.append(self.query2id[item])
        return output_list

    def get_linked_text(self, title):
        """get linked text"""
        linked_title_list = []
        raw_title_list = self.data_db[get_new_title(title)]['linked_title'].split('\t')
        for item in raw_title_list:
            if item and self.data_db[get_new_title(item)].get("text"):
                linked_title_list.append(get_new_title(item))
        output_twohop_list = []
        for item in linked_title_list:
            output_twohop_list.append(self.data_db[get_new_title(item)]['text'])
        return output_twohop_list, linked_title_list

    def convert_onehop_to_features(self, query,
                                   cls_token='[CLS]',
                                   sep_token='[SEP]',
                                   pad_token=0):
        """convert one hop data to features"""
        query_id = self.get_query2id(query)
        examples = []
        count = 0
        for item in query_id:
            title_doc_list = []
            for i in range(len(self.q_doc_text[item]['text'][:self.num_docs])):
                title_doc_list.append([query[count], self.q_doc_text[item]["text"][i]])
            examples += title_doc_list
            count += 1

        max_q_len = self.max_q_len
        max_doc_len = self.max_doc_len
        tokenizer = BertTokenizer.from_pretrained(self.vocab, do_lower_case=True)
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        for _, example in enumerate(examples):
            tokens_q = tokenizer.tokenize(example[0])
            tokens_d1 = tokenizer.tokenize(example[1])
            special_tokens_count = 2
            if len(tokens_q) > max_q_len - 1:
                tokens_q = tokens_q[:(max_q_len - 1)]
            if len(tokens_d1) > max_doc_len - special_tokens_count:
                tokens_d1 = tokens_d1[:(max_doc_len - special_tokens_count)]
            tokens_q = [cls_token] + tokens_q
            tokens_d = [sep_token]
            tokens_d += tokens_d1
            tokens_d += [sep_token]

            q_ids = tokenizer.convert_tokens_to_ids(tokens_q)
            d_ids = tokenizer.convert_tokens_to_ids(tokens_d)
            padding_length_d = max_doc_len - len(d_ids)
            padding_length_q = max_q_len - len(q_ids)
            input_ids = q_ids + ([pad_token] * padding_length_q) + d_ids + ([pad_token] * padding_length_d)
            token_type_ids = [0] * max_q_len
            token_type_ids += [1] * max_doc_len
            attention_mask_id = []

            for item in input_ids:
                attention_mask_id.append(item != 0)

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask_id)
        return input_ids_list, token_type_ids_list, attention_mask_list

    def convert_twohop_to_features(self, examples,
                                   cls_token='[CLS]',
                                   sep_token='[SEP]',
                                   pad_token=0):
        """convert two hop data to features"""
        max_q_len = self.max_q_len
        max_doc_len = self.max_doc_len
        max_seq_len = self.max_seq_len2
        tokenizer = BertTokenizer.from_pretrained(self.vocab, do_lower_case=True)
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []

        for _, example in enumerate(examples):
            tokens_q = tokenizer.tokenize(example[0])
            tokens_d1 = tokenizer.tokenize(example[1])
            tokens_d2 = tokenizer.tokenize(example[2])

            special_tokens_count1 = 1
            special_tokens_count2 = 2

            if len(tokens_q) > max_q_len - 1:
                tokens_q = tokens_q[:(max_q_len - 1)]
            if len(tokens_d1) > max_doc_len - special_tokens_count1:
                tokens_d1 = tokens_d1[:(max_doc_len - special_tokens_count1)]
            if len(tokens_d2) > max_doc_len - special_tokens_count2:
                tokens_d2 = tokens_d2[:(max_doc_len - special_tokens_count2)]
            tokens = [cls_token] + tokens_q
            tokens += [sep_token]
            tokens += tokens_d1
            tokens += [sep_token]
            tokens += tokens_d2
            tokens += [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)

            token_type_ids = [0] * (len(tokens_q) + 1)
            token_type_ids += [1] * (len(tokens_d1) + 1)
            token_type_ids += [1] * (max_seq_len - len(token_type_ids))

            attention_mask_id = []
            for item in input_ids:
                attention_mask_id.append(item != 0)

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask_id)
        return input_ids_list, token_type_ids_list, attention_mask_list

    def get_samples(self, query, onehop_index, onehop_prob):
        """get samples"""
        query = self.get_query2id([query])
        index_np = onehop_index.asnumpy()
        onehop_prob = onehop_prob.asnumpy()
        sample = []
        path = []
        last_out = []
        q_id = query[0]
        q_text = self.id2query[q_id]
        onehop_ids_list = index_np

        onehop_text_list = []
        onehop_title_list = []
        for ids in list(onehop_ids_list):
            onehop_text_list.append(self.id2doc[q_id][ids])
            onehop_title_list.append(self.id2title[q_id][ids])
        twohop_text_list = []
        twohop_title_list = []
        for title in onehop_title_list:
            two_hop_text, two_hop_title = self.get_linked_text(title)
            twohop_text_list.append(two_hop_text[:1000])
            twohop_title_list.append(two_hop_title[:1000])
        d1_count = 0
        d2_count = 0
        tiny_sample = []
        tiny_path = []
        for i in range(1, self.onehop_num):
            tiny_sample.append((q_text, onehop_text_list[0], onehop_text_list[i]))
            tiny_path.append((get_new_title(onehop_title_list[0]), get_new_title(onehop_title_list[i])))
            last_out.append(onehop_prob[d1_count])
        for twohop_text_tiny_list in twohop_text_list:
            for twohop_text in twohop_text_tiny_list:
                tiny_sample.append((q_text, onehop_text_list[d1_count], twohop_text))
                last_out.append(onehop_prob[d1_count])
            d1_count += 1
        for twohop_title_tiny_list in twohop_title_list:
            for twohop_title in twohop_title_tiny_list:
                tiny_path.append((get_new_title(onehop_title_list[d2_count]), get_new_title(twohop_title)))
            d2_count += 1
        sample += tiny_sample
        path += tiny_path
        return sample, path, last_out
