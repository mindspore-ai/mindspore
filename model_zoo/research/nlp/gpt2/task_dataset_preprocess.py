# Copyright 2020 Huawei Technologies Co., Ltd
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
"""dataset preprocess"""

import argparse

from src.utils.data_preprocess import lambada_dataset_preprocess
from src.utils.data_preprocess import cbt_dataset_preprocess
from src.utils.data_preprocess import wikitext_dataset_preprocess
from src.utils.data_preprocess import ptb_dataset_preprocess
from src.utils.data_preprocess import onebw_dataset_preprocess
from src.utils.data_preprocess import coqa_dataset_preprocess
from src.utils.data_preprocess import wmt14_en_fr_preprocess


def main():
    parser = argparse.ArgumentParser(description="All Task dataset preprocessing")
    parser.add_argument("--task", type=str, default="translation",
                        help="The GPT-2 downstream task, including [LanguageModeling, CBT, Translation, Lambada"
                             "Summarization, ReadingComprehension]")
    parser.add_argument("--input_file", type=str, default="",
                        help="The raw dataset path. ")
    parser.add_argument("--dataset", type=str, default="onebw",
                        help="The name of dataset which should be processed, only for LanguageModeling task.")
    parser.add_argument("--output_file", type=str, default="",
                        help="The output dataset path after preprocessing.")
    parser.add_argument("--condition", type=str, default="test",
                        help="Process train or test dataset, including [train, test], only for 1BW and "
                             "CNN & DailyMail dataset.")
    args_opt = parser.parse_args()

    task = args_opt.task
    condition = args_opt.condition
    dataset = args_opt.dataset
    input_file = args_opt.input_file
    output_file = args_opt.output_file

    if task.lower() == "languagemodeling":
        print("Start processing Language Modeling dataset ...")
        if dataset.lower() == "wikitext2" or dataset.lower() == "wikitext103":
            wikitext_dataset_preprocess(input_file=input_file, output_file=output_file)
        elif dataset.lower() == "ptb":
            ptb_dataset_preprocess(input_file=input_file, output_file=output_file)
        elif dataset.lower() == "onebw":
            onebw_dataset_preprocess(condition, input_file=input_file, output_file=output_file)
        else:
            raise ValueError("Only support wikitext2, wikitext103, ptb, onebw dataset")

    elif task.lower() == "lambada":
        print("Start processing Lambada dataset ...")
        lambada_dataset_preprocess(input_file=input_file, output_file=output_file)

    elif task.lower() == "cbt":
        print("Start processing CBT dataset ...")
        cbt_dataset_preprocess(input_file=input_file, output_file=output_file)

    elif task.lower() == "readingcomprehension":
        print("Start processing ReadingComprehension dataset ...")
        coqa_dataset_preprocess(input_file=input_file, output_file=output_file)

    elif task.lower() == "summarization":
        print("Start processing Summarization dataset ...")

    elif task.lower() == "translation":
        print("Start processing Translation dataset ...")
        wmt14_en_fr_preprocess(input_file=input_file, output_file=output_file)

    else:
        raise ValueError("Only support Language Modeling, CBT, Translation, Lambada, "
                         "Summarization, Reading Comprehension task.")


if __name__ == "__main__":
    main()
