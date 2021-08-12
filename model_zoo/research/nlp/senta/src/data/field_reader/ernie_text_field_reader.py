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
:py:class:`ErnieTextFieldReader`

"""

from src.common.register import RegisterSet
from src.data.util_helper import pad_batch_data
from src.utils.util_helper import truncation_words


@RegisterSet.field_reader.register
class ErnieTextFieldReader():
    '''
    ErnieTextFieldReader
    '''

    def __init__(self, field_config):
        """
        :param field_config:
        """
        self.field_config = field_config
        self.tokenizer = None
        self.token_embedding = None

        if self.field_config.tokenizer_info:
            tokenizer_class = RegisterSet.tokenizer.__getitem__(self.field_config.tokenizer_info["type"])
            params = None
            if self.field_config.tokenizer_info.__contains__("params"):
                params = self.field_config.tokenizer_info["params"]
            self.tokenizer = tokenizer_class(vocab_file=self.field_config.vocab_path,
                                             split_char=self.field_config.tokenizer_info["split_char"],
                                             unk_token=self.field_config.tokenizer_info["unk_token"],
                                             params=params)

    def convert_texts_to_ids(self, batch_text):
        """
        :param batch_text:
        :return:
        """
        src_ids = []
        position_ids = []
        task_ids = []
        sentence_ids = []
        for text in batch_text:
            if self.field_config.need_convert:
                tokens_text = self.tokenizer.tokenize(text)
                if len(tokens_text) > self.field_config.max_seq_len - 2:
                    tokens_text = truncation_words(tokens_text, self.field_config.max_seq_len - 2,
                                                   self.field_config.truncation_type)
                tokens = []
                tokens.append("[CLS]")
                for token in tokens_text:
                    tokens.append(token)
                tokens.append("[SEP]")
                src_id = self.tokenizer.convert_tokens_to_ids(tokens)
            else:
                if isinstance(text, str):
                    src_id = text.split(" ")
                src_id = [int(i) for i in text]
                if len(src_id) > self.field_config.max_seq_len - 2:
                    src_id = truncation_words(src_id, self.field_config.max_seq_len - 2,
                                              self.field_config.truncation_type)
                src_id.insert(0, self.tokenizer.covert_token_to_id("[CLS]"))
                src_id.append(self.tokenizer.covert_token_to_id("[SEP]"))

            src_ids.append(src_id)
            pos_id = list(range(len(src_id)))
            task_id = [0] * len(src_id)
            sentence_id = [0] * len(src_id)
            position_ids.append(pos_id)
            task_ids.append(task_id)
            sentence_ids.append(sentence_id)

        return_list = []
        padded_ids, input_mask = pad_batch_data(src_ids,
                                                pad_idx=self.field_config.padding_id,
                                                return_input_mask=True,
                                                return_seq_lens=True)
        sent_ids_batch = pad_batch_data(sentence_ids, pad_idx=self.field_config.padding_id)
        pos_ids_batch = pad_batch_data(position_ids, pad_idx=self.field_config.padding_id)

        return_list.append(padded_ids)  # append src_ids
        return_list.append(sent_ids_batch)  # append sent_ids
        return_list.append(pos_ids_batch)  # append pos_ids
        return_list.append(input_mask)  # append mask

        return return_list
