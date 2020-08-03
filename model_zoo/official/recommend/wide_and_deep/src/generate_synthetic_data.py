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

"""Generate the synthetic data for wide&deep model training"""
import time
import argparse
import numpy as np


def generate_data(output_path, label_dim, number_examples, dense_dim, slot_dim, vocabulary_size, random_slot_values):
    """
    This function generates the synthetic data of the web clicking data. Each row in the output file is as follows
    'label\tdense_feature[0] dense_feature[1] ... sparse_feature[0]...sparse_feature[1]...'
    Each value is dilimited by '\t'.
    Args:
          output_path: string. The output file path of the synthetic data
          label_dim: int. The category of the label. For 0-1 clicking problem, it's value is 2
          number_examples: int. The row numbers of the synthetic dataset
          dense_dim: int. The number of continue features.
          slot_dim: int. The number of the category features
          vocabulary_size: int. The value of vocabulary size
          random_slot_values: bool. If true, the id is geneted by the random. If false, the id is set by the row_index
                            mod part_size, where part_size the the vocab size for each slot
    """

    part_size = (vocabulary_size - dense_dim) // slot_dim

    if random_slot_values is True:
        print('Each field size is supposed to be {}, so number of examples should be no less than this value'.format(
            part_size))

    start = time.time()

    buffer_data = []

    with open(output_path, 'w') as fp:
        for i in range(number_examples):
            example = []
            label = i % label_dim
            example.append(label)

            dense_feature = ["{:.3f}".format(j + 0.01 * i % 10) for j in range(dense_dim)]
            example.extend(dense_feature)

            if random_slot_values is True:
                for j in range(slot_dim):
                    example.append(dense_dim + np.random.randint(j * part_size, min((j + 1) * part_size,
                                                                                    vocabulary_size - dense_dim - 1)))
            else:
                sp = i % part_size
                example.extend(
                    [dense_dim + min(sp + j * part_size, vocabulary_size - dense_dim - 1) for j in range(slot_dim)])

            buffer_data.append("\t".join([str(item) for item in example]))

            if (i + 1) % 10000 == 0:
                end = time.time()
                speed = 10000 / (end - start)
                start = time.time()
                print("Processed {} examples with speed {:.2f} examples/s".format(i + 1, speed))
                fp.write('\n'.join(buffer_data) + '\n')
                buffer_data = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Synthetic Data')

    parser.add_argument("--output_file", type=str, default="./train.txt", help='The output path of the generated file')
    parser.add_argument("--label_dim", type=int, default=2, help='The label category')
    parser.add_argument("--number_examples", type=int, default=4000000, help='The row numbers of the generated file')
    parser.add_argument("--dense_dim", type=int, default=13, help='The number of the continue feature.')
    parser.add_argument("--slot_dim", type=int, default=26, help="The number of the category features")
    parser.add_argument("--vocabulary_size", type=int, default=400000000,
                        help="The vocabulary size of the total dataset")
    parser.add_argument("--random_slot_values", type=int, default=0,
                        help="If 1, the id is geneted by the random. If false, the id is set by "
                             "the row_index mod part_size, where part_size the the vocab size for each slot")
    args = parser.parse_args()
    args.random_slot_values = bool(args.random_slot_values)

    generate_data(output_path=args.output_file, label_dim=args.label_dim, number_examples=args.number_examples,
                  dense_dim=args.dense_dim, slot_dim=args.slot_dim, vocabulary_size=args.vocabulary_size,
                  random_slot_values=args.random_slot_values)
