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

'''
Bert preprocess script.
'''

import os
import argparse
from src.dataset import create_ner_dataset


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="bert preprocess")
    parser.add_argument("--assessment_method", type=str, default="BF1", choices=["BF1", "clue_benchmark", "MF1"],
                        help="assessment_method include: [BF1, clue_benchmark, MF1], default is BF1")
    parser.add_argument("--do_eval", type=str, default="false", choices=["true", "false"],
                        help="Eable eval, default is false")
    parser.add_argument("--use_crf", type=str, default="false", choices=["true", "false"],
                        help="Use crf, default is false")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--vocab_file_path", type=str, default="", help="Vocab file path, used in clue benchmark")
    parser.add_argument("--label_file_path", type=str, default="", help="label file path, used in clue benchmark")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--dataset_format", type=str, default="mindrecord", choices=["mindrecord", "tfrecord"],
                        help="Dataset format, support mindrecord or tfrecord")
    parser.add_argument("--schema_file_path", type=str, default="",
                        help="Schema path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')

    args_opt = parser.parse_args()

    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    if args_opt.assessment_method.lower() == "clue_benchmark" and args_opt.vocab_file_path == "":
        raise ValueError("'vocab_file_path' must be set to do clue benchmark")
    if args_opt.use_crf.lower() == "true" and args_opt.label_file_path == "":
        raise ValueError("'label_file_path' must be set to use crf")
    if args_opt.assessment_method.lower() == "clue_benchmark" and args_opt.label_file_path == "":
        raise ValueError("'label_file_path' must be set to do clue benchmark")
    if args_opt.assessment_method.lower() == "clue_benchmark":
        args_opt.eval_batch_size = 1
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    assessment_method = args.assessment_method.lower()
    if args.do_eval.lower() == "true":
        ds = create_ner_dataset(batch_size=args.eval_batch_size, repeat_count=1,
                                assessment_method=assessment_method, data_file_path=args.eval_data_file_path,
                                schema_file_path=args.schema_file_path, dataset_format=args.dataset_format,
                                do_shuffle=(args.eval_data_shuffle.lower() == "true"), drop_remainder=False)
        ids_path = os.path.join(args.result_path, "00_data")
        mask_path = os.path.join(args.result_path, "01_data")
        token_path = os.path.join(args.result_path, "02_data")
        label_path = os.path.join(args.result_path, "03_data")
        os.makedirs(ids_path)
        os.makedirs(mask_path)
        os.makedirs(token_path)
        os.makedirs(label_path)

        for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
            input_ids = data["input_ids"]
            input_mask = data["input_mask"]
            token_type_id = data["segment_ids"]
            label_ids = data["label_ids"]

            file_name = "cluener_bs" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"
            ids_file_path = os.path.join(ids_path, file_name)
            input_ids.tofile(ids_file_path)

            mask_file_path = os.path.join(mask_path, file_name)
            input_mask.tofile(mask_file_path)

            token_file_path = os.path.join(token_path, file_name)
            token_type_id.tofile(token_file_path)

            label_file_path = os.path.join(label_path, file_name)
            label_ids.tofile(label_file_path)
        print("=" * 20, "export bin files finished", "=" * 20)
