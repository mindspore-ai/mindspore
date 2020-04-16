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
# ==============================================================================
import mindspore.dataset as ds
from mindspore import log as logger
import mindspore.dataset.transforms.nlp.utils as nlp

DATA_FILE = "../data/dataset/testTextFileDataset/1.txt"
DATA_ALL_FILE = "../data/dataset/testTextFileDataset/*"

def test_textline_dataset_one_file():
    data = ds.TextFileDataset(DATA_FILE)
    count = 0
    for i in data.create_dict_iterator():
        logger.info("{}".format(i["text"]))
        count += 1
    assert(count == 3)

def test_textline_dataset_all_file():
    data = ds.TextFileDataset(DATA_ALL_FILE)
    count = 0
    for i in data.create_dict_iterator():
        logger.info("{}".format(i["text"]))
        count += 1
    assert(count == 5)

def test_textline_dataset_totext():
    data = ds.TextFileDataset(DATA_ALL_FILE, shuffle=False)
    count = 0
    line = ["This is a text file.", "Another file.", "Be happy every day.", "End of file.", "Good luck to everyone."]
    for i in data.create_dict_iterator():
        str = nlp.as_text(i["text"])
        assert(str == line[count])       
        count += 1
    assert(count == 5)

def test_textline_dataset_num_samples():
    data = ds.TextFileDataset(DATA_FILE, num_samples=2)
    count = 0
    for i in data.create_dict_iterator():
        count += 1
    assert(count == 2)

def test_textline_dataset_distribution():
    data = ds.TextFileDataset(DATA_ALL_FILE, num_shards=2, shard_id=1)
    count = 0
    for i in data.create_dict_iterator():
        count += 1
    assert(count == 3)

def test_textline_dataset_repeat():
    data = ds.TextFileDataset(DATA_FILE, shuffle=False)
    data = data.repeat(3)
    count = 0
    line = ["This is a text file.", "Be happy every day.", "Good luck to everyone.",
            "This is a text file.", "Be happy every day.", "Good luck to everyone.",
            "This is a text file.", "Be happy every day.", "Good luck to everyone."]
    for i in data.create_dict_iterator():
        str = nlp.as_text(i["text"])
        assert(str == line[count])       
        count += 1
    assert(count == 9)

def test_textline_dataset_get_datasetsize():
    data = ds.TextFileDataset(DATA_FILE)
    size = data.get_dataset_size()
    assert(size == 3)

if __name__ == "__main__":
    test_textline_dataset_one_file()
    test_textline_dataset_all_file()
    test_textline_dataset_totext()
    test_textline_dataset_num_samples()
    test_textline_dataset_distribution()
    test_textline_dataset_repeat()
    test_textline_dataset_get_datasetsize()
