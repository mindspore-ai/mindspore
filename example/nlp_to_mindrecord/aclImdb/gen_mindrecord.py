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
"""get data from aclImdb and write the data to mindrecord file"""
import os
from mindspore.mindrecord import FileWriter

ACLIMDB_DIR = "data/aclImdb"

MINDRECORD_FILE_NAME_TRAIN = "output/aclImdb_train.mindrecord"
MINDRECORD_FILE_NAME_TEST = "output/aclImdb_test.mindrecord"

def get_data_as_dict(data_dir):
    """get data from dir like aclImdb/train"""
    dir_list = [os.path.join(data_dir, "pos"),
                os.path.join(data_dir, "neg")]

    for index, exact_dir in enumerate(dir_list):
        if not os.path.exists(exact_dir):
            raise IOError("dir {} not exists".format(exact_dir))

        for item in os.listdir(exact_dir):
            data = {}
            data["label"] = int(index)    # indicate pos: 0, neg: 1

            # file name like 4372_2.txt, we will get id: 4372, score: 2
            id_score = item.split("_", 1)
            score = id_score[1].split(".", 1)
            data["id"] = int(id_score[0])
            data["score"] = int(score[0])

            review_file = open(os.path.join(exact_dir, item), "r")
            review = review_file.read()
            review_file.close()
            data["review"] = str(review)
            yield data

def gen_mindrecord(data_type):
    """gen mindreocrd according exactly schema"""
    if data_type == "train":
        fw = FileWriter(MINDRECORD_FILE_NAME_TRAIN)
    else:
        fw = FileWriter(MINDRECORD_FILE_NAME_TEST)

    schema = {"id": {"type": "int32"},
              "label": {"type": "int32"},
              "score": {"type": "int32"},
              "review": {"type": "string"}}
    fw.add_schema(schema, "aclImdb dataset")
    fw.add_index(["id", "label", "score"])

    get_data_iter = get_data_as_dict(os.path.join(ACLIMDB_DIR, data_type))

    batch_size = 256
    transform_count = 0
    while True:
        data_list = []
        try:
            for _ in range(batch_size):
                data_list.append(get_data_iter.__next__())
                transform_count += 1
            fw.write_raw_data(data_list)
            print(">> transformed {} record...".format(transform_count))
        except StopIteration:
            if data_list:
                fw.write_raw_data(data_list)
                print(">> transformed {} record...".format(transform_count))
            break

    fw.commit()

def main():
    # generate mindrecord for train
    print(">> begin generate mindrecord by train data")
    gen_mindrecord("train")

    # generate mindrecord for test
    print(">> begin generate mindrecord by test data")
    gen_mindrecord("test")

if __name__ == "__main__":
    main()
