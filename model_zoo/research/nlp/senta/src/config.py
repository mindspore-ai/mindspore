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
'''config'''
import mindspore.common.dtype as mstype
from src.bert_model import BertConfig

bertcfg = BertConfig(
    seq_length=512,
    vocab_size=50265,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=1024 * 4,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
    use_relative_positions=False,
    dtype=mstype.float32,
    compute_type=mstype.float16)

sstcfg = {
    "dataset_reader": {
        "train_reader": {
            "name": "train_reader",
            "type": "OneSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json"
                    }
                }
            ],
            "config": {
                "data_path": "../data/en/finetune/SST-2/train",
                "shuffle": True,
                "batch_size": 1,
                "epoch": 10,
                "sampling_rate": 1.0
            }
        },
        "test_reader": {
            "name": "test_reader",
            "type": "OneSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json",
                        "other": ""
                    }
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                }
            ],
            "config": {
                "data_path": "../data/en/finetune/SST-2/test",
                "shuffle": False,
                "batch_size": 1,
                "epoch": 1,
                "sampling_rate": 1.0
            }
        },
        "dev_reader": {
            "name": "dev_reader",
            "type": "OneSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json",
                        "other": ""
                    }
                }
            ],
            "config": {
                "data_path": "../data/en/finetune/SST-2/dev",
                "shuffle": False,
                "batch_size": 1,
                "epoch": 1,
                "sampling_rate": 1.0
            }
        }
    },
    "model": {
        "type": "RobertaOneSentClassificationEn",
        "embedding": {
            "type": "ErnieTokenEmbedding",
            "emb_dim": 1024,
            "use_fp16": False,
            "config_path": "./model_files/config/roberta_large_en.config.json",
            "other": ""
        },
        "optimization": {
            "learning_rate": 3e-5,
        }
    }

}

semcfg = {
    "dataset_reader": {
        "train_reader": {
            "name": "train_reader",
            "type": "RobertaTwoSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json"
                    }
                },
                {
                    "name": "text_b",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json"
                    }
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                }
            ],
            "config": {
                "data_path": "../data/en/finetune/absa_laptops/train",
                "shuffle": True,
                "batch_size": 1,
                "epoch": 10,
                "sampling_rate": 1.0,
                "extra_params": {
                    "vocab_path": "../roberta_en.vocab.txt",
                    "bpe_vocab_file": "../roberta_en.vocab.bpe",
                    "bpe_json_file": "../roberta_en.encoder.json",
                    "label_map_config": "",
                    "max_seq_len": 512,
                    "do_lower_case": True,
                    "in_tokens": False,
                    "is_classify": True,
                    "tokenizer": "GptBpeTokenizer",
                    "data_augmentation": False,
                    "text_field_more_than_3": False,
                    "is_regression": False,
                    "use_multi_gpu_test": True
                }
            }
        },
        "test_reader": {
            "name": "test_reader",
            "type": "RobertaTwoSentClassifyReaderEn",
            "fields": [
                {
                    "name": "qid",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                },
                {
                    "name": "text_a",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json",
                        "other": ""
                    }
                },
                {
                    "name": "text_b",
                    "data_type": "string",
                    "reader": {"type": "ErnieTextFieldReader"},
                    "tokenizer": {
                        "type": "GptBpeTokenizer",
                        "split_char": " ",
                        "unk_token": "[UNK]",
                        "params": {
                            "bpe_vocab_file": "../roberta_en.vocab.bpe",
                            "bpe_json_file": "../roberta_en.encoder.json"
                        }
                    },
                    "need_convert": True,
                    "vocab_path": "../roberta_en.vocab.txt",
                    "max_seq_len": 512,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": {
                        "type": "ErnieTokenEmbedding",
                        "use_reader_emb": True,
                        "emb_dim": 1024,
                        "config_path": "./model_files/config/roberta_large_en.config.json",
                        "other": ""
                    }
                },
                {
                    "name": "label",
                    "data_type": "int",
                    "reader": {"type": "ScalarFieldReader"},
                    "tokenizer": None,
                    "need_convert": False,
                    "vocab_path": "",
                    "max_seq_len": 1,
                    "truncation_type": 0,
                    "padding_id": 1,
                    "embedding": None
                }
            ],
            "config": {
                "data_path": "../data/en/finetune/absa_laptops/test",
                "shuffle": False,
                "batch_size": 1,
                "epoch": 1,
                "sampling_rate": 1.0,
                "extra_params": {
                    "vocab_path": "../roberta_en.vocab.txt",
                    "bpe_vocab_file": "../roberta_en.vocab.bpe",
                    "bpe_json_file": "../roberta_en.encoder.json",
                    "label_map_config": "",
                    "max_seq_len": 512,
                    "do_lower_case": True,
                    "in_tokens": False,
                    "is_classify": True,
                    "tokenizer": "GptBpeTokenizer",
                    "data_augmentation": False,
                    "text_field_more_than_3": False,
                    "is_regression": False,
                    "use_multi_gpu_test": True
                }
            }
        }
    },
    "model": {
        "type": "RobertaOneSentClassificationEn",
        "embedding": {
            "type": "ErnieTokenEmbedding",
            "emb_dim": 1024,
            "use_fp16": False,
            "config_path": "./model_files/config/roberta_large_en.config.json",
            "other": ""
        },
        "optimization": {
            "learning_rate": 3e-5,
        }
    }
}
