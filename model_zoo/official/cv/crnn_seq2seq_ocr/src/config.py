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
#" ============================================================================
"""Config parameters for CRNN-Seq2Seq-OCR model."""

from easydict import EasyDict as ed


config = ed({

    # dataset-related
    "mindrecord_dir": "",
    "data_root": "",
    "annotation_file": "",

    "val_data_root": "",
    "val_annotation_file": "",
    "data_json": "",

    "characters_dictionary": {"pad_id": 0, "go_id": 1, "eos_id": 2, "unk_id": 3},
    "labels_not_use": [u'%#�?%', u'%#背景#%', u'%#不识�?%', u'#%不识�?#', u'%#模糊#%', u'%#模糊#%'],
    "vocab_path": "./general_chars.txt",

    #model-related
    "img_width": 512,
    "img_height": 128,
    "channel_size": 3,
    "conv_out_dim": 384,
    "encoder_hidden_size": 128,
    "decoder_hidden_size": 128,
    "decoder_output_size": 10000, # vocab_size is the decoder_output_size, characters_class+1, last 9999 is the space
    "dropout_p": 0.1,
    "max_length": 64,
    "attn_num_layers": 1,
    "teacher_force_ratio": 0.5,

    #optimizer-related
    "lr": 0.0008,
    "adam_beta1": 0.5,
    "adam_beta2": 0.999,
    "loss_scale": 1024,

    #train-related
    "batch_size": 32,
    "num_epochs": 20,
    "keep_checkpoint_max": 20,

    #eval-related
    "eval_batch_size": 32
})
