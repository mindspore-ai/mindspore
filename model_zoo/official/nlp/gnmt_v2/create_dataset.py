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
"""Create Dataset."""
import os
import argparse

from src.dataset.bi_data_loader import BiLingualDataLoader, TextDataLoader
from src.dataset.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='Generate dataset file.')
parser.add_argument("--src_folder", type=str, default="/home/workspace/wmt16_de_en", required=False,
                    help="Raw corpus folder.")

parser.add_argument("--output_folder", type=str, default="/home/workspace/dataset_menu",
                    required=False,
                    help="Dataset output path.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    dicts = []
    train_src_file = "train.tok.clean.bpe.32000.en"
    train_tgt_file = "train.tok.clean.bpe.32000.de"
    test_src_file = "newstest2014.en"
    test_tgt_file = "newstest2014.de"

    vocab = args.src_folder + "/vocab.bpe.32000"
    bpe_codes = args.src_folder + "/bpe.32000"
    pad_vocab = 8
    tokenizer = Tokenizer(vocab, bpe_codes, src_en='en', tgt_de='de', vocab_pad=pad_vocab)

    test = TextDataLoader(
        src_filepath=os.path.join(args.src_folder, test_src_file),
        tokenizer=tokenizer,
        source_max_sen_len=None,
        schema_address=args.output_folder + "/" + test_src_file + ".json"
    )
    print(f" | It's writing, please wait a moment.")
    test.write_to_mindrecord(
        path=os.path.join(
            args.output_folder,
            os.path.basename(test_src_file) + ".mindrecord"
        ),
        train_mode=False
    )

    train = BiLingualDataLoader(
        src_filepath=os.path.join(args.src_folder, train_src_file),
        tgt_filepath=os.path.join(args.src_folder, train_tgt_file),
        tokenizer=tokenizer,
        source_max_sen_len=51,
        target_max_sen_len=50,
        schema_address=args.output_folder + "/" + train_src_file + ".json"
    )
    print(f" | It's writing, please wait a moment.")
    train.write_to_mindrecord(
        path=os.path.join(
            args.output_folder,
            os.path.basename(train_src_file) + ".mindrecord"
        ),
        train_mode=True
    )

    print(f" | Vocabulary size: {tokenizer.vocab_size}.")
