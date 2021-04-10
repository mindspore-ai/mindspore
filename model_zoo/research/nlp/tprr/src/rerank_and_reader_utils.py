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
"""utils"""

import re
import argparse
from urllib.parse import unquote
from collections import defaultdict
import collections
import logging
import unicodedata
import json
import gzip
import string
import pickle
import sqlite3
import numpy as np
from tqdm import tqdm

from transformers import BasicTokenizer


logger = logging.getLogger(__name__)


class Example:
    """A single example of data"""
    def __init__(self,
                 qas_id,
                 path,
                 unique_id,
                 question_tokens,
                 doc_tokens,
                 sent_names,
                 sup_fact_id,
                 sup_para_id,
                 para_start_end_position,
                 sent_start_end_position,
                 question_text,
                 title_start_end_position=None):
        """init function"""
        self.qas_id = qas_id
        self.path = path
        self.unique_id = unique_id
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.sup_para_id = sup_para_id
        self.para_start_end_position = para_start_end_position
        self.sent_start_end_position = sent_start_end_position
        self.title_start_end_position = title_start_end_position


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 qas_id,
                 path,
                 sent_names,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 para_spans,
                 sent_spans,
                 token_to_orig_map):
        """init function"""
        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids
        self.path = path
        self.unique_id = unique_id
        self.sent_names = sent_names

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.para_spans = para_spans
        self.sent_spans = sent_spans

        self.token_to_orig_map = token_to_orig_map


class DocDB:
    """
    Sqlite backed document storage.
    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path):
        """init function"""
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        """enter function"""
        return self

    def __exit__(self, *args):
        """exit function"""
        self.close()

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_info(self, doc_id):
        """get docment information"""
        if not doc_id.endswith('_0'):
            doc_id += '_0'
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE id = ?",
            (normalize_title(doc_id),)
        )
        result = cursor.fetchall()
        cursor.close()
        return result if result is None else result[0]


def get_parse():
    """get parse function"""
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--data_path',
                        type=str,
                        default="",
                        help='data path')
    parser.add_argument('--ckpt_path',
                        type=str,
                        default="",
                        help='ckpt path')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--seq_len', type=int, default=512,
                        help="max sentence length")
    parser.add_argument("--get_reranker_data",
                        action='store_true',
                        help="Set this flag if you want to get reranker data from retrieved result")
    parser.add_argument("--run_reranker",
                        action='store_true',
                        help="Set this flag if you want to run reranker")
    parser.add_argument("--cal_reranker_metrics",
                        action='store_true',
                        help="Set this flag if you want to calculate rerank metrics")
    parser.add_argument("--select_reader_data",
                        action='store_true',
                        help="Set this flag if you want to select reader data")
    parser.add_argument("--run_reader",
                        action='store_true',
                        help="Set this flag if you want to run reader")
    parser.add_argument("--cal_reader_metrics",
                        action='store_true',
                        help="Set this flag if you want to calculate reader metrics")
    parser.add_argument('--dev_gold_file',
                        type=str,
                        default="hotpot_dev_fullwiki_v1.json",
                        help='file of dev ground truth')
    parser.add_argument('--wiki_db_file',
                        type=str,
                        default="enwiki_offset.db",
                        help='wiki_database_file')
    parser.add_argument('--albert_model',
                        type=str,
                        default="albert-xxlarge",
                        help='model path of huggingface albert-xxlarge')

    # Retriever
    parser.add_argument('--retriever_result_file',
                        type=str,
                        default="../doc_path",
                        help='file of retriever result')

    # Rerank
    parser.add_argument('--rerank_batch_size', type=int, default=32,
                        help="rerank batchsize for evaluating")
    parser.add_argument('--rerank_feature_file',
                        type=str,
                        default="../reranker_feature_file.pkl.gz",
                        help='file of rerank feature')
    parser.add_argument('--rerank_example_file',
                        type=str,
                        default="../reranker_example_file.pkl.gz",
                        help='file of rerank example')
    parser.add_argument('--rerank_result_file',
                        type=str,
                        default="../rerank_result.json",
                        help='file of rerank result')
    parser.add_argument('--rerank_encoder_ck_file',
                        type=str,
                        default="rerank_albert.ckpt",
                        help='checkpoint of rerank albert-xxlarge')
    parser.add_argument('--rerank_downstream_ck_file',
                        type=str,
                        default="rerank_downstream.ckpt",
                        help='checkpoint of rerank downstream')

    # Reader
    parser.add_argument('--reader_batch_size', type=int, default=32,
                        help="reader batchsize for evaluating")
    parser.add_argument('--reader_feature_file',
                        type=str,
                        default="../reader_feature_file.pkl.gz",
                        help='file of reader feature')
    parser.add_argument('--reader_example_file',
                        type=str,
                        default="../reader_example_file.pkl.gz",
                        help='file of reader example')
    parser.add_argument('--reader_encoder_ck_file',
                        type=str,
                        default="reader_albert.ckpt",
                        help='checkpoint of reader albert-xxlarge')
    parser.add_argument('--reader_downstream_ck_file',
                        type=str,
                        default="reader_downstream.ckpt",
                        help='checkpoint of reader downstream')
    parser.add_argument('--reader_result_file',
                        type=str,
                        default="../reader_result_file.json",
                        help='file of reader result')
    parser.add_argument('--sp_threshold', type=float, default=0.65,
                        help="threshold for selecting supporting sentences")
    parser.add_argument("--max_para_num", default=2, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)

    return parser


def select_reader_dev_data(args):
    """select reader dev data from result of retriever based on result of reranker"""
    rerank_result_file = args.rerank_result_file
    rerank_feature_file = args.rerank_feature_file
    rerank_example_file = args.rerank_example_file
    reader_feature_file = args.reader_feature_file
    reader_example_file = args.reader_example_file

    with gzip.open(rerank_example_file, "rb") as f:
        dev_examples = pickle.load(f)
    with gzip.open(rerank_feature_file, "rb") as f:
        dev_features = pickle.load(f)
    with open(rerank_result_file, "r") as f:
        rerank_result = json.load(f)

    new_dev_examples = []
    new_dev_features = []

    rerank_unique_ids = defaultdict(int)
    feature_unique_ids = defaultdict(int)

    for _, res in tqdm(rerank_result.items(), desc="get rerank unique ids"):
        rerank_unique_ids[res[0]] = True

    for feature in tqdm(dev_features, desc="select rerank top1 feature"):
        if feature.unique_id in rerank_unique_ids:
            feature_unique_ids[feature.unique_id] = True
            new_dev_features.append(feature)

    for example in tqdm(dev_examples, desc="select rerank top1 example"):
        if example.unique_id in rerank_unique_ids and example.unique_id in feature_unique_ids:
            new_dev_examples.append(example)

    with gzip.open(reader_example_file, "wb") as f:
        pickle.dump(new_dev_examples, f)

    with gzip.open(reader_feature_file, "wb") as f:
        pickle.dump(new_dev_features, f)
    print("finish selecting reader data !!!")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def get_ans_from_pos(tokenizer, examples, features, y1, y2, unique_id):
    """get answer text from predicted position"""
    feature = features[unique_id]
    example = examples[unique_id]
    tok_to_orig_map = feature.token_to_orig_map
    orig_all_tokens = example.question_tokens + example.doc_tokens

    final_text = " "
    if y1 < len(tok_to_orig_map) and y2 < len(tok_to_orig_map):
        orig_tok_start = tok_to_orig_map[y1]
        orig_tok_end = tok_to_orig_map[y2]
        # -----------------orig all tokens-----------------------------------
        orig_tokens = orig_all_tokens[orig_tok_start: (orig_tok_end + 1)]
        tok_tokens = feature.doc_tokens[y1: y2 + 1]
        tok_text = tokenizer.convert_tokens_to_string(tok_tokens)
        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)
        final_text = get_final_text(tok_text, orig_text, True, False)
    return final_text


def convert_to_tokens(examples, features, ids, y1, y2, q_type_prob, tokenizer, sent, sent_names,
                      unique_ids):
    """get raw answer text and supporting sentences"""
    answer_dict = defaultdict(list)

    q_type = np.argmax(q_type_prob, 1)

    for i, qid in enumerate(ids):
        unique_id = unique_ids[i]

        if q_type[i] == 0:
            answer_text = 'yes'
        elif q_type[i] == 1:
            answer_text = 'no'
        elif q_type[i] == 2:
            answer_text = get_ans_from_pos(tokenizer, examples, features, y1[i], y2[i], unique_id)
        else:
            raise ValueError("question type error")

        answer_dict[qid].append(answer_text)
        answer_dict[qid].append(sent[i])
        answer_dict[qid].append(sent_names[i])

    return answer_dict


def normalize_title(text):
    """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
    text = unicodedata.normalize('NFD', text)
    return text[0].capitalize() + text[1:]


def make_wiki_id(title, para_index):
    """make wiki id"""
    title_id = "{0}_{1}".format(normalize_title(title), para_index)
    return title_id


def cal_reranker_metrics(dev_gold_file, rerank_result_file):
    """function for calculating reranker's metrics"""
    with open(dev_gold_file, 'rb') as f:
        gt = json.load(f)
    with open(rerank_result_file, 'rb') as f:
        rerank_result = json.load(f)

    cnt = 0
    all_ = len(gt)

    cnt_c = 0
    cnt_b = 0
    all_c = 0
    all_b = 0

    for item in tqdm(gt, desc="get com and bridge "):
        q_type = item["type"]
        if q_type == "comparison":
            all_c += 1
        elif q_type == "bridge":
            all_b += 1
        else:
            print(f"{q_type} is a error question type!!!")

    for item in tqdm(gt, desc="cal pem"):
        _id = item["_id"]
        if _id in rerank_result:
            pred = rerank_result[_id][1]
            sps = item["supporting_facts"]
            q_type = item["type"]
            gold = []
            for t in sps:
                gold.append(normalize_title(t[0]))
            gold = set(gold)
            flag = True
            for t in gold:
                if t not in pred:
                    flag = False
                    break
            if flag:
                cnt += 1
                if q_type == "comparison":
                    cnt_c += 1
                elif q_type == "bridge":
                    cnt_b += 1
                else:
                    print(f"{q_type} is a error question type!!!")

    return cnt/all_, cnt_c/all_c, cnt_b/all_b


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def find_hyper_linked_titles(text_w_links):
    """find hyperlinked titles"""
    titles = re.findall(r'href=[\'"]?([^\'" >]+)', text_w_links)
    titles = [unquote(title) for title in titles]
    titles = [title[0].capitalize() + title[1:] for title in titles]
    return titles


def normalize_text(text):
    """Resolve different type of unicode encodings / capitarization in HotpotQA data."""
    text = unicodedata.normalize('NFD', text)
    return text


def convert_char_to_token_offset(orig_text, start_offset, end_offset, char_to_word_offset, doc_tokens):
    """build characters' offset"""
    length = len(orig_text)
    assert start_offset + length == end_offset
    assert end_offset <= len(char_to_word_offset)

    start_position = char_to_word_offset[start_offset]
    end_position = char_to_word_offset[start_offset + length - 1]

    actual_text = " ".join(
        doc_tokens[start_position:(end_position + 1)])

    assert actual_text.lower().find(orig_text.lower()) != -1
    return start_position, end_position


def _is_whitespace(c):
    """check whitespace"""
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def convert_text_to_tokens(context_text, return_word_start=False):
    """convert text to tokens"""
    doc_tokens = []
    char_to_word_offset = []
    words_start_idx = []
    prev_is_whitespace = True

    for idx, c in enumerate(context_text):
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
                words_start_idx.append(idx)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    if not return_word_start:
        return doc_tokens, char_to_word_offset
    return doc_tokens, char_to_word_offset, words_start_idx


def read_json(eval_file_name):
    """reader json files"""
    print("loading examples from {0}".format(eval_file_name))
    with open(eval_file_name) as reader:
        lines = json.load(reader)
    return lines


def write_json(data, out_file_name):
    """write json files"""
    print("writing {0} examples to {1}".format(len(data), out_file_name))
    with open(out_file_name, 'w') as writer:
        json.dump(data, writer, indent=4)


def get_edges(sentence):
    """get edges"""
    EDGE_XY = re.compile(r'<a href="(?!http|<a)(.*?)">(.*?)<\/a>')
    ret = EDGE_XY.findall(sentence)
    return [(unquote(x), y) for x, y in ret]


def relocate_tok_span(orig_to_tok_index, orig_to_tok_back_index, word_tokens, subword_tokens,
                      orig_start_position, orig_end_position, orig_text, tokenizer, tok_to_orig_index=None):
    """relocate tokens' span"""
    if orig_start_position is None:
        return 0, 0

    tok_start_position = orig_to_tok_index[orig_start_position]
    if tok_start_position >= len(subword_tokens):
        return 0, 0

    if orig_end_position < len(word_tokens) - 1:
        tok_end_position = orig_to_tok_back_index[orig_end_position]
        if tok_to_orig_index and tok_to_orig_index[tok_end_position + 1] == -1:
            assert tok_end_position <= orig_to_tok_index[orig_end_position + 1] - 2
        else:
            assert tok_end_position == orig_to_tok_index[orig_end_position + 1] - 1
    else:
        tok_end_position = orig_to_tok_back_index[orig_end_position]
    return _improve_answer_span(
        subword_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)


def generate_mapping(length, positions):
    """generate mapping"""
    start_mapping = [0] * length
    end_mapping = [0] * length
    for _, (start, end) in enumerate(positions):
        start_mapping[start] = 1
        end_mapping[end] = 1
    return start_mapping, end_mapping


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text, add_prefix_space=True))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _largest_valid_index(spans, limit):
    """return largest valid index"""
    for idx, _ in enumerate(spans):
        if spans[idx][1] >= limit:
            return idx
    return len(spans)


def remove_punc(text):
    """remove punctuation"""
    if text == " ":
        return ''
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def check_text_include_ans(ans, text):
    """check whether text include answer"""
    if normalize_answer(ans) in normalize_answer(text):
        return True
    return False


def remove_articles(text):
    """remove articles"""
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    """fix whitespace"""
    return ' '.join(text.split())


def lower(text):
    """lower text"""
    return text.lower()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(lower(s))))
