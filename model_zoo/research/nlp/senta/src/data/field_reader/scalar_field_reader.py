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
:py:class:`ScalarFieldReader`

"""
import numpy as np

from src.common.register import RegisterSet
from src.data.tokenizer.custom_tokenizer import CustomTokenizer


@RegisterSet.field_reader.register
class ScalarFieldReader():
    """
    return shape= [batch_size,1]
    """
    def __init__(self, field_config):
        """
        :param field_config:
        """
        self.field_config = field_config
        self.tokenizer = None
        self.token_embedding = None

        if field_config.vocab_path and field_config.need_convert:
            self.tokenizer = CustomTokenizer(vocab_file=self.field_config.vocab_path)


    def convert_texts_to_ids(self, batch_text):
        """
        :param batch_text:
        :return:
        """
        src_ids = []
        for text in batch_text:
            src_id = text.split(" ")
            if self.tokenizer and self.field_config.need_convert:
                scalar = self.tokenizer.covert_token_to_id(src_id[0])
            else:
                scalar = src_id[0]
            src_ids.append(scalar)

        return_list = []
        if self.field_config.data_type == 'float':
            return_list.append(np.array(src_ids).astype("float32").reshape([-1, 1]))

        elif self.field_config.data_type == 'int':
            return_list.append(np.array(src_ids).astype("int32").reshape([-1, 1]))

        return return_list
