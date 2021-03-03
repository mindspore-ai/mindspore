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
# ==============================================================================
import mindspore.dataset as ds
from mindspore import log as logger


def test_num_samples():
    manifest_file = "../data/dataset/testManifestData/test5trainimgs.json"
    num_samples = 1
    # sampler = ds.DistributedSampler(num_shards=1, shard_id=0, shuffle=False, num_samples=3, offset=1)
    data1 = ds.ManifestDataset(
        manifest_file, num_samples=num_samples, num_shards=3, shard_id=1
    )
    row_count = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        row_count += 1
    assert row_count == 1


def test_num_samples_tf():
    logger.info("test_tfrecord_read_all_dataset")
    schema_file = "../data/dataset/testTFTestAllTypes/datasetSchemaNoRow.json"
    files = ["../data/dataset/testTFTestAllTypes/test.data"]
    # here num samples indicate the rows per shard. Total rows in file = 12
    ds1 = ds.TFRecordDataset(files, schema_file, num_samples=2)
    count = 0
    for _ in ds1.create_tuple_iterator(num_epochs=1):
        count += 1
    assert count == 2


def test_num_samples_image_folder():
    data_dir = "../data/dataset/testPK/data"
    ds1 = ds.ImageFolderDataset(data_dir, num_samples=2, num_shards=2, shard_id=0)
    count = 0
    for _ in ds1.create_tuple_iterator(num_epochs=1):
        count += 1
    assert count == 2


if __name__ == "__main__":
    test_num_samples()
    test_num_samples_tf()
    test_num_samples_image_folder()
