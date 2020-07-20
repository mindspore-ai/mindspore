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
"""Create pre-training dataset."""
import os
from multiprocessing import Pool, cpu_count

from src.dataset import MonoLingualDataLoader
from src.language_model import LooseMaskedLanguageModel


def _create_pre_train(text_file, vocabulary, output_folder_path,
                      mask_ratio,
                      mask_all_prob,
                      min_sen_len,
                      max_sen_len,
                      suffix,
                      dataset_type):
    """
    Create pre-training dataset.

    Args:
        text_file (str): Text file path.
        vocabulary (Dictionary): Vocab instance.
        output_folder_path (str): Output folder path.
        mask_ratio (float): Mask ratio.
        mask_all_prob (float): Mask all ratio.
        min_sen_len (int): Minimum sentence length.
        max_sen_len (int): Maximum sentence length.
        suffix (str): Suffix of output file.
        dataset_type (str): Tfrecord or mindrecord.
    """
    suffix = suffix if not suffix else "_" + suffix
    loader = MonoLingualDataLoader(
        src_filepath=text_file,
        lang="en", dictionary=vocabulary,
        language_model=LooseMaskedLanguageModel(mask_ratio=mask_ratio, mask_all_prob=mask_all_prob),
        max_sen_len=max_sen_len, min_sen_len=min_sen_len
    )
    src_file_name = os.path.basename(text_file)
    if dataset_type.lower() == "tfrecord":
        file_name = os.path.join(
            output_folder_path,
            src_file_name.replace('.txt', f'_len_{max_sen_len}{suffix}.tfrecord')
        )
        loader.write_to_tfrecord(path=file_name)
    else:
        file_name = os.path.join(
            output_folder_path,
            src_file_name.replace('.txt', f'_len_{max_sen_len}{suffix}.mindrecord')
        )
        loader.write_to_mindrecord(path=file_name)


def create_pre_training_dataset(folder_path,
                                output_folder_path,
                                vocabulary,
                                prefix, suffix="",
                                mask_ratio=0.3,
                                mask_all_prob=None,
                                min_sen_len=7,
                                max_sen_len=82,
                                dataset_type="tfrecord",
                                cores=2):
    """
    Create pre-training dataset.

    Args:
        folder_path (str): Text file folder path.
        vocabulary (Dictionary): Vocab instance.
        output_folder_path (str): Output folder path.
        mask_ratio (float): Mask ratio.
        mask_all_prob (float): Mask all ratio.
        min_sen_len (int): Minimum sentence length.
        max_sen_len (int): Maximum sentence length.
        prefix (str): Prefix of text file.
        suffix (str): Suffix of output file.
        dataset_type (str): Tfrecord or mindrecord.
        cores (int): Cores to use.
    """
    # Second step of data preparation.
    # Create mono zh-zh train MindRecord.
    if not os.path.exists(output_folder_path):
        raise NotADirectoryError(f"`output_folder_path` is not existed.")
    if not os.path.isdir(output_folder_path):
        raise NotADirectoryError(f"`output_folder_path` must be a dir.")

    data_file = []
    dirs = os.listdir(folder_path)
    for file in dirs:
        if file.startswith(prefix) and file.endswith(".txt"):
            data_file.append(os.path.join(folder_path, file))

    if not data_file:
        raise FileNotFoundError("No available text file found.")

    args_groups = []
    for text_file in data_file:
        args_groups.append((text_file,
                            vocabulary,
                            output_folder_path,
                            mask_ratio,
                            mask_all_prob,
                            min_sen_len,
                            max_sen_len,
                            suffix,
                            dataset_type))

    cores = min(cores, cpu_count())
    pool = Pool(cores)
    for arg in args_groups:
        pool.apply_async(_create_pre_train, args=arg)
    pool.close()
    pool.join()

    print(f" | Generate Dataset for Pre-training is done.")
