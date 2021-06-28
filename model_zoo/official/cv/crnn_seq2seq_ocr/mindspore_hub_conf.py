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
"""hub config"""
from src.attention_ocr import AttentionOCRInfer
from src.config import config

def crnnseq2seqocr_net(*args, **kwargs):
    return AttentionOCRInfer(*args, **kwargs)

def create_network(name, *args, **kwargs):
    """create_network about crnn_seq2seq_ocr"""
    if name == "crnn_seq2seq_ocr":
        return crnnseq2seqocr_net(config.batch_size,
                                  int(config.img_width / 4),
                                  config.encoder_hidden_size,
                                  config.decoder_hidden_size,
                                  config.decoder_output_size,
                                  config.max_length,
                                  config.dropout_p,
                                  *args,
                                  **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
