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

"""download the CNN & DailyMail for Summarization task"""

import argparse
from datasets import load_dataset


def generate_txt(url, split_, number=None, version="3.0.0"):
    """
    generate txt file of cnn_dailymail dataset

    Args:
        url (str): directory of dataset txt file.
        split_ (str): test or train.
        number (int): top-n number of samples from dataset
        version (str): "3.0.0" by default

    """
    cnn = load_dataset("cnn_dailymail", version, split=split_)
    if number == -1:
        number = len(cnn)
    f = open(url + split_ + '.txt', 'w')
    for idx in range(number):
        article = cnn[idx]['article']
        article = article.replace('\n', ' ')
        highlights = cnn[idx]['highlights']
        highlights = highlights.replace('\n', ' ')
        f.write(article + "\t" + highlights + '\n')
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download CNN_Dailymail 3.0.0 using datasets by Huggingface')
    parser.add_argument('--dir', type=str, default="", help="directory of dataset")
    parser.add_argument('--split', type=str, default='test', help="[test,train]")
    parser.add_argument('--num', type=int, default=-1,
                        help=" number of samples by default order. "
                             "If num is -1, it will download whole dataset. Default: -1")
    args = parser.parse_args()

    data_directory = args.dir
    split = args.split
    num = args.num

    generate_txt(url=data_directory, split_=split, number=num)
