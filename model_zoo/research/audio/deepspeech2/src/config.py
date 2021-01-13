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
# ===========================================================================
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

train_config = ed({

    "TrainingConfig": {
        "epochs": 70,
    },

    "DataConfig": {
        "train_manifest": 'data/libri_train_manifest.csv',
        # "val_manifest": 'data/libri_val_manifest.csv',
        "batch_size": 20,
        "labels_path": "labels.json",

        "SpectConfig": {
            "sample_rate": 16000,
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hamming"
        },

        "AugmentationConfig": {
            "speed_volume_perturb": False,
            "spec_augment": False,
            "noise_dir": '',
            "noise_prob": 0.4,
            "noise_min": 0.0,
            "noise_max": 0.5,
        }
    },

    "ModelConfig": {
        "rnn_type": "LSTM",
        "hidden_size": 1024,
        "hidden_layers": 5,
        "lookahead_context": 20,
    },

    "OptimConfig": {
        "learning_rate": 3e-4,
        "learning_anneal": 1.1,
        "weight_decay": 1e-5,
        "momentum": 0.9,
        "eps": 1e-8,
        "betas": (0.9, 0.999),
        "loss_scale": 1024,
        "epsilon": 0.00001
    },

    "CheckpointConfig": {
        "ckpt_file_name_prefix": 'DeepSpeech',
        "ckpt_path": './checkpoint',
        "keep_checkpoint_max": 10
    }
})

eval_config = ed({

    "save_output": 'librispeech_val_output',
    "verbose": True,

    "DataConfig": {
        "test_manifest": 'data/libri_test_clean_manifest.csv',
        # "test_manifest": 'data/libri_test_other_manifest.csv',
        # "test_manifest": 'data/libri_val_manifest.csv',
        "batch_size": 20,
        "labels_path": "labels.json",

        "SpectConfig": {
            "sample_rate": 16000,
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hanning"
        },
    },

    "ModelConfig": {
        "rnn_type": "LSTM",
        "hidden_size": 1024,
        "hidden_layers": 5,
        "lookahead_context": 20,
    },

    "LMConfig": {
        "decoder_type": "greedy",
        "lm_path": './3-gram.pruned.3e-7.arpa',
        "top_paths": 1,
        "alpha": 1.818182,
        "beta": 0,
        "cutoff_top_n": 40,
        "cutoff_prob": 1.0,
        "beam_width": 1024,
        "lm_workers": 4
    },

})
