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
"""build reranker data from retriever result"""
import pickle
import gzip
from tqdm import tqdm

from src.rerank_and_reader_utils import read_json, make_wiki_id, convert_text_to_tokens, normalize_title, \
    whitespace_tokenize, DocDB, _largest_valid_index, generate_mapping, InputFeatures, Example

from transformers import AutoTokenizer


def judge_para(data):
    """judge whether is valid para"""
    for _, para_tokens in data["context"].items():
        if len(para_tokens) == 1:
            return False
    return True


def judge_sp(data, sent_name2id, para2id):
    """judge whether is valid sp"""
    for sp in data['sp']:
        title = normalize_title(sp[0])
        name = normalize_title(sp[0]) + '_{}'.format(sp[1])
        if title in para2id and name not in sent_name2id:
            return False
    return True


def judge(path, path_set, reverse=False, golds=None, mode='or'):
    """judge function"""
    if path[0] == path[-1]:
        return False
    if path in path_set:
        return False
    if reverse and path[::-1] in path_set:
        return False
    if not golds:
        return True
    if mode == 'or':
        return any(gold not in path for gold in golds)
    if mode == 'and':
        return all(gold not in path for gold in golds)
    return False


def get_context_and_sents(path, doc_db):
    """get context ans sentences"""
    context = {}
    sents = {}
    for title in path:
        para_info = doc_db.get_doc_info(title)
        if title.endswith('_0'):
            title = title[:-2]
        context[title] = pickle.loads(para_info[1])
        sents[title] = pickle.loads(para_info[2])
    return context, sents


def gen_dev_data(dev_file, db_path, topk_file):
    """generate dev data"""
    # ----------------------------------------db info-----------------------------------------------
    topk_data = read_json(topk_file)  # path
    doc_db = DocDB(db_path)  # db get offset
    print('load db successfully!')

    # ---------------------------------------------supervision ------------------------------------------
    dev_data = read_json(dev_file)
    qid2sp = {}
    qid2ans = {}
    qid2type = {}
    qid2path = {}
    for _, data in enumerate(dev_data):
        sp_facts = data['supporting_facts'] if 'supporting_facts' in data else None
        qid2sp[data['_id']] = sp_facts
        qid2ans[data['_id']] = data['answer'] if 'answer' in data else None
        qid2type[data['_id']] = data['type'] if 'type' in data else None
        qid2path[data['_id']] = list(set(list(zip(*sp_facts))[0])) if sp_facts else None

    new_dev_data = []

    for _, data in enumerate(tqdm(topk_data)):
        qid = data['q_id']
        question = data['question']
        topk_titles = data['topk_titles']
        gold_path = list(map(normalize_title, qid2path[qid])) if qid2path[qid] else None

        all_titles = []
        for titles in topk_titles:
            titles = list(map(normalize_title, titles))
            if len(titles) == 1:
                continue
            path = titles[:2]
            if judge(path, all_titles):
                all_titles.append(titles[:2])
            if len(titles) == 3:
                path = titles[1:]
                if judge(path, all_titles):
                    all_titles.append(titles[1:])

        # --------------------------------------------------process query-----------------------------------

        question = " ".join(whitespace_tokenize(question))
        question = question.strip()
        q_tokens, _ = convert_text_to_tokens(question)

        gold_path = list(map(lambda x: make_wiki_id(x, 0), gold_path)) if gold_path else None
        for path in all_titles:
            context, sents = get_context_and_sents(path, doc_db)
            ans_label = int(gold_path[0] in path and gold_path[1] in path) if gold_path else None

            new_dev_data.append({
                'qid': qid,
                'type': qid2type[qid],
                'question': question,
                'q_tokens': q_tokens,
                'context': context,
                'sents': sents,
                'answer': qid2ans[qid],
                'sp': qid2sp[qid],
                'ans_para': None,
                'is_impossible': not ans_label == 1
            })

    return new_dev_data


def read_hotpot_examples(path_data):
    """reader examples"""
    examples = []
    max_sent_cnt = 0
    failed = 0

    for _, data in enumerate(path_data):
        if not judge_para(data):
            failed += 1
            continue
        question = data['question']
        question = " ".join(whitespace_tokenize(question))
        question = question.strip()
        path = list(map(normalize_title, data["context"].keys()))
        qid = data['qid']
        q_tokens = data['q_tokens']

        # -------------------------------------add para------------------------------------------------------------
        doc_tokens = []

        para_start_end_position = []
        title_start_end_position = []
        sent_start_end_position = []

        sent_names = []
        sent_name2id = {}
        para2id = {}

        for para, para_tokens in data["context"].items():
            sents = data["sents"][para]

            para = normalize_title(para)
            title_tokens = convert_text_to_tokens(para)[0]
            para_node_id = len(para_start_end_position)
            para2id[para] = para_node_id

            doc_offset = len(doc_tokens)
            doc_tokens += title_tokens
            doc_tokens += para_tokens

            title_start_end_position.append((doc_offset, doc_offset + len(title_tokens) - 1))

            doc_offset += len(title_tokens)
            para_start_end_position.append((doc_offset, doc_offset + len(para_tokens) - 1, para))

            for idx, sent in enumerate(sents):
                if sent[0] == -1 and sent[1] == -1:
                    continue
                sent_names.append([para, idx])  # local name
                sent_node_id = len(sent_start_end_position)
                sent_name2id[normalize_title(para) + '_{}'.format(str(idx))] = sent_node_id
                sent_start_end_position.append((doc_offset + sent[0],
                                                doc_offset + sent[1]))

        # add sp and ans
        sp_facts = []
        sup_fact_id = []
        for sp in sp_facts:
            name = normalize_title(sp[0]) + '_{}'.format(sp[1])
            if name in sent_name2id:
                sup_fact_id.append(sent_name2id[name])

        sup_para_id = set()  # use set
        if sp_facts:
            for para in list(zip(*sp_facts))[0]:
                para = normalize_title(para)
                if para in para2id:
                    sup_para_id.add(para2id[para])
        sup_para_id = list(sup_para_id)

        example = Example(
            qas_id=qid,
            path=path,
            unique_id=qid + '_' + '_'.join(path),
            question_tokens=q_tokens,
            doc_tokens=doc_tokens,  # multi-para tokens w/o query
            sent_names=sent_names,
            sup_fact_id=sup_fact_id,  # global sent id
            sup_para_id=sup_para_id,  # global para id
            para_start_end_position=para_start_end_position,
            sent_start_end_position=sent_start_end_position,
            title_start_end_position=title_start_end_position,
            question_text=question)

        examples.append(example)
        max_sent_cnt = max(max_sent_cnt, len(sent_start_end_position))

    print(f"Maximum sentence cnt: {max_sent_cnt}")
    print(f'failed examples: {failed}')
    print(f'convert {len(examples)} examples successfully!')

    return examples


def add_sub_token(sub_tokens, idx, tok_to_orig_index, all_query_tokens):
    """add sub tokens"""
    for sub_token in sub_tokens:
        tok_to_orig_index.append(idx)
        all_query_tokens.append(sub_token)
    return tok_to_orig_index, all_query_tokens


def get_sent_spans(example, orig_to_tok_index, orig_to_tok_back_index):
    """get sentences' spans"""
    sentence_spans = []
    for sent_span in example.sent_start_end_position:
        sent_start_position = orig_to_tok_index[sent_span[0]]
        sent_end_position = orig_to_tok_back_index[sent_span[1]]
        sentence_spans.append((sent_start_position, sent_end_position + 1))
    return sentence_spans


def get_para_spans(example, orig_to_tok_index, orig_to_tok_back_index, all_doc_tokens, marker):
    """get paragraphs' spans"""
    para_spans = []
    for title_span, para_span in zip(example.title_start_end_position, example.para_start_end_position):
        para_start_position = orig_to_tok_index[title_span[0]]
        para_end_position = orig_to_tok_back_index[para_span[1]]
        if para_end_position + 1 < len(all_doc_tokens) and all_doc_tokens[para_end_position + 1] == \
                marker['sent'][0]:
            para_spans.append((para_start_position - 1, para_end_position + 1, para_span[2]))
        else:
            para_spans.append((para_start_position - 1, para_end_position, para_span[2]))
    return para_spans


def build_feature(example, all_doc_tokens, doc_input_ids, doc_input_mask, doc_segment_ids, all_query_tokens,
                  query_input_ids, query_input_mask, query_segment_ids, para_spans, sentence_spans, tok_to_orig_index):
    """build a input feature"""
    feature = InputFeatures(
                qas_id=example.qas_id,
                path=example.path,
                unique_id=example.qas_id + '_' + '_'.join(example.path),
                sent_names=example.sent_names,
                doc_tokens=all_doc_tokens,
                doc_input_ids=doc_input_ids,
                doc_input_mask=doc_input_mask,
                doc_segment_ids=doc_segment_ids,
                query_tokens=all_query_tokens,
                query_input_ids=query_input_ids,
                query_input_mask=query_input_mask,
                query_segment_ids=query_segment_ids,
                para_spans=para_spans,
                sent_spans=sentence_spans,
                token_to_orig_map=tok_to_orig_index)
    return feature


def convert_example_to_features(tokenizer, args, examples):
    """convert examples to features"""
    features = []
    failed = 0
    marker = {'q': ['[q]', '[/q]'], 'para': ['<t>', '</t>'], 'sent': ['[s]']}

    for (_, example) in enumerate(tqdm(examples)):

        all_query_tokens = [tokenizer.cls_token, marker['q'][0]]
        tok_to_orig_index = [-1, -1]  # orig: query + doc tokens
        ques_orig_to_tok_index = []  # start position
        ques_orig_to_tok_back_index = []  # end position
        q_spans = []

        # -------------------------------------------for query---------------------------------------------
        for (idx, token) in enumerate(example.question_tokens):
            sub_tokens = tokenizer.tokenize(token)

            ques_orig_to_tok_index.append(len(all_query_tokens))
            tok_to_orig_index, all_query_tokens = add_sub_token(sub_tokens, idx, tok_to_orig_index, all_query_tokens)
            ques_orig_to_tok_back_index.append(len(all_query_tokens) - 1)

        all_query_tokens = all_query_tokens[:63]
        tok_to_orig_index = tok_to_orig_index[:63]
        all_query_tokens.append(marker['q'][-1])
        tok_to_orig_index.append(-1)
        q_spans.append((1, len(all_query_tokens) - 1))

        # ---------------------------------------add doc tokens------------------------------------------------
        all_doc_tokens = []
        orig_to_tok_index = []  # orig: token in doc
        orig_to_tok_back_index = []
        title_start_mapping, title_end_mapping = generate_mapping(len(example.doc_tokens),
                                                                  example.title_start_end_position)
        _, sent_end_mapping = generate_mapping(len(example.doc_tokens),
                                               example.sent_start_end_position)
        all_doc_tokens += all_query_tokens

        for (idx, token) in enumerate(example.doc_tokens):
            sub_tokens = tokenizer.tokenize(token)

            if title_start_mapping[idx] == 1:
                all_doc_tokens.append(marker['para'][0])
                tok_to_orig_index.append(-1)

            # orig: position in doc tokens tok: global tokenized tokens (start)
            orig_to_tok_index.append(len(all_doc_tokens))
            tok_to_orig_index, all_doc_tokens = add_sub_token(sub_tokens, idx + len(example.question_tokens),
                                                              tok_to_orig_index, all_doc_tokens)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

            if title_end_mapping[idx] == 1:
                all_doc_tokens.append(marker['para'][1])
                tok_to_orig_index.append(-1)

            if sent_end_mapping[idx] == 1:
                all_doc_tokens.append(marker['sent'][0])
                tok_to_orig_index.append(-1)

        # -----------------------------------for sentence-------------------------------------------------
        sentence_spans = get_sent_spans(example, orig_to_tok_index, orig_to_tok_back_index)

        # -----------------------------------for para-------------------------------------------------------
        para_spans = get_para_spans(example, orig_to_tok_index, orig_to_tok_back_index, all_doc_tokens, marker)

        # -----------------------------------remove sent > max seq length-----------------------------------------
        sent_max_index = _largest_valid_index(sentence_spans, args.seq_len)
        max_sent_cnt = len(sentence_spans)

        if sent_max_index != len(sentence_spans):
            if sent_max_index == 0:
                failed += 0
                continue
            sentence_spans = sentence_spans[:sent_max_index]
            max_tok_length = sentence_spans[-1][1]  # max_tok_length [s]

            # max end index: max_tok_length
            para_max_index = _largest_valid_index(para_spans, max_tok_length + 1)
            if para_max_index == 0:  # only one para
                failed += 0
                continue
            if orig_to_tok_back_index[example.title_start_end_position[1][1]] + 1 >= max_tok_length:
                failed += 0
                continue
            max_para_span = para_spans[para_max_index]
            para_spans = para_spans[:para_max_index]
            para_spans.append((max_para_span[0], max_tok_length, max_para_span[2]))

            all_doc_tokens = all_doc_tokens[:max_tok_length + 1]

        sentence_spans = sentence_spans[:min(max_sent_cnt, args.max_sent_num)]

        # ----------------------------------------Padding Document-----------------------------------------------------
        if len(all_doc_tokens) > args.seq_len:
            st, _, title = para_spans[-1]
            para_spans[-1] = (st, args.seq_len - 1, title)
            all_doc_tokens = all_doc_tokens[:args.seq_len - 1] + [marker['sent'][0]]

        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        query_input_ids = tokenizer.convert_tokens_to_ids(all_query_tokens)

        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(query_input_ids) + [1] * (len(doc_input_ids) - len(query_input_ids))

        doc_pad_length = args.seq_len - len(doc_input_ids)
        doc_input_ids += [0] * doc_pad_length
        doc_input_mask += [0] * doc_pad_length
        doc_segment_ids += [0] * doc_pad_length

        # Padding Question
        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        query_pad_length = 64 - len(query_input_ids)
        query_input_ids += [0] * query_pad_length
        query_input_mask += [0] * query_pad_length
        query_segment_ids += [0] * query_pad_length

        feature = build_feature(example, all_doc_tokens, doc_input_ids, doc_input_mask, doc_segment_ids,
                                all_query_tokens, query_input_ids, query_input_mask, query_segment_ids, para_spans,
                                sentence_spans, tok_to_orig_index)
        features.append(feature)
    return features


def get_rerank_data(args):
    """function for generating reranker's data"""
    new_dev_data = gen_dev_data(dev_file=args.dev_gold_path,
                                db_path=args.wiki_db_path,
                                topk_file=args.retriever_result_file)
    tokenizer = AutoTokenizer.from_pretrained(args.albert_model_path)
    new_tokens = ['[q]', '[/q]', '<t>', '</t>', '[s]']
    tokenizer.add_tokens(new_tokens)

    examples = read_hotpot_examples(new_dev_data)
    features = convert_example_to_features(tokenizer=tokenizer, args=args, examples=examples)

    with gzip.open(args.rerank_example_file, "wb") as f:
        pickle.dump(examples, f)
    with gzip.open(args.rerank_feature_file, "wb") as f:
        pickle.dump(features, f)
