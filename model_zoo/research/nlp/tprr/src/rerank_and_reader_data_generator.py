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
"""define a data generator"""

import gzip
import pickle
import random
import numpy as np


random.seed(42)
np.random.seed(42)


class DataGenerator:
    """data generator for reranker and reader"""
    def __init__(self, feature_file_path, example_file_path, batch_size, seq_len,
                 para_limit=None, sent_limit=None, task_type=None):
        """init function"""
        self.example_ptr = 0
        self.bsz = batch_size
        self.seq_length = seq_len
        self.para_limit = para_limit
        self.sent_limit = sent_limit
        self.task_type = task_type

        self.feature_file_path = feature_file_path
        self.example_file_path = example_file_path
        self.features = self.load_features()
        self.examples = self.load_examples()
        self.feature_dict = self.get_feature_dict()
        self.example_dict = self.get_example_dict()

        self.features = self.padding_feature(self.features, self.bsz)

    def load_features(self):
        """load features from feature file"""
        with gzip.open(self.feature_file_path, 'rb') as fin:
            features = pickle.load(fin)
            print("load features successful !!!")
            return features

    def padding_feature(self, features, bsz):
        """padding features as multiples of batch size"""
        padding_num = ((len(features) // bsz + 1) * bsz - len(features))
        print(f"features padding num is {padding_num}")
        new_features = features + features[:padding_num]
        return new_features

    def load_examples(self):
        """laod examples from file"""
        if self.example_file_path:
            with gzip.open(self.example_file_path, 'rb') as fin:
                examples = pickle.load(fin)
                print("load examples successful !!!")
                return examples
        return {}

    def get_feature_dict(self):
        """build a feature dict"""
        return {f.unique_id: f for f in self.features}

    def get_example_dict(self):
        """build a example dict"""
        if self.example_file_path:
            return {e.unique_id: e for e in self.examples}
        return {}

    def common_process_single_case(self, i, case, context_idxs, context_mask, segment_idxs, ids, path, unique_ids):
        """common process for a single case"""
        context_idxs[i] = np.array(case.doc_input_ids)
        context_mask[i] = np.array(case.doc_input_mask)
        segment_idxs[i] = np.array(case.doc_segment_ids)

        ids.append(case.qas_id)
        path.append(case.path)
        unique_ids.append(case.unique_id)

        return context_idxs, context_mask, segment_idxs, ids, path, unique_ids

    def reader_process_single_case(self, i, case, sent_names, square_mask, query_mapping, ques_start_mapping,
                                   para_start_mapping, sent_end_mapping):
        """process for a single case about reader"""
        sent_names.append(case.sent_names)
        prev_position = None
        for cur_position, token_id in enumerate(case.doc_input_ids):
            if token_id >= 30000:
                if prev_position:
                    square_mask[i, prev_position + 1: cur_position, prev_position + 1: cur_position] = 1.0
                prev_position = cur_position
        if case.sent_spans:
            for j in range(case.sent_spans[0][0] - 1):
                query_mapping[i, j] = 1
        ques_start_mapping[i, 0, 1] = 1
        for j, para_span in enumerate(case.para_spans[:self.para_limit]):
            start, end, _ = para_span
            if start <= end:
                para_start_mapping[i, j, start] = 1
        for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
            start, end = sent_span
            if start <= end:
                end = min(end, self.seq_length - 1)
                sent_end_mapping[i, j, end] = 1
        return sent_names, square_mask, query_mapping, ques_start_mapping, para_start_mapping, sent_end_mapping

    def __iter__(self):
        """iteration function"""
        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            # BERT input
            context_idxs = np.zeros((cur_bsz, self.seq_length))
            context_mask = np.zeros((cur_bsz, self.seq_length))
            segment_idxs = np.zeros((cur_bsz, self.seq_length))

            # others
            ids = []
            path = []
            unique_ids = []

            if self.task_type == "reader":
                # Mappings
                ques_start_mapping = np.zeros((cur_bsz, 1, self.seq_length))
                query_mapping = np.zeros((cur_bsz, self.seq_length))
                para_start_mapping = np.zeros((cur_bsz, self.para_limit, self.seq_length))
                sent_end_mapping = np.zeros((cur_bsz, self.sent_limit, self.seq_length))
                square_mask = np.zeros((cur_bsz, self.seq_length, self.seq_length))
                sent_names = []

            for i, case in enumerate(cur_batch):
                context_idxs, context_mask, segment_idxs, ids, path, unique_ids = \
                    self.common_process_single_case(i, case, context_idxs, context_mask, segment_idxs, ids, path,
                                                    unique_ids)
                if self.task_type == "reader":
                    sent_names, square_mask, query_mapping, ques_start_mapping, para_start_mapping, sent_end_mapping = \
                        self.reader_process_single_case(i, case, sent_names, square_mask, query_mapping,
                                                        ques_start_mapping, para_start_mapping, sent_end_mapping)

            self.example_ptr += cur_bsz

            if self.task_type == "reranker":
                yield {
                    "context_idxs": context_idxs,
                    "context_mask": context_mask,
                    "segment_idxs": segment_idxs,

                    "ids": ids,
                    "unique_ids": unique_ids,
                    "path": path
                }
            elif self.task_type == "reader":
                yield {
                    "context_idxs": context_idxs,
                    "context_mask": context_mask,
                    "segment_idxs": segment_idxs,
                    "query_mapping": query_mapping,
                    "para_start_mapping": para_start_mapping,
                    "sent_end_mapping": sent_end_mapping,
                    "square_mask": square_mask,
                    "ques_start_mapping": ques_start_mapping,

                    "ids": ids,
                    "unique_ids": unique_ids,
                    "sent_names": sent_names,
                    "path": path
                }
            else:
                print(f"data generator received a error type: {self.task_type} !!!")
