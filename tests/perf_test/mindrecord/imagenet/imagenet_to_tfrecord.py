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
"""generate tfrecord"""
import collections
import os

import tensorflow as tf

IMAGENET_MAP_FILE = "../../../ut/data/mindrecord/testImageNetDataWhole/labels_map.txt"
IMAGENET_IMAGE_DIR = "../../../ut/data/mindrecord/testImageNetDataWhole/images"
TFRECORD_FILE = "./imagenet.tfrecord"
PARTITION_NUMBER = 16


def get_imagenet_filename_label_pic(map_file, image_dir):
    """
    Get data from imagenet.

    Yields:
        filename, label, image_bytes
    """
    if not os.path.exists(map_file):
        raise IOError("map file {} not exists".format(map_file))

    label_dict = {}
    with open(map_file) as fp:
        line = fp.readline()
        while line:
            labels = line.split(" ")
            label_dict[labels[1]] = labels[0]
            line = fp.readline()

    # get all the dir which are n02087046, n02094114, n02109525
    dir_paths = {}
    for item in label_dict:
        real_path = os.path.join(image_dir, label_dict[item])
        if not os.path.isdir(real_path):
            print("{} dir is not exist".format(real_path))
            continue
        dir_paths[item] = real_path

    if not dir_paths:
        raise PathNotExistsError("not valid image dir in {}".format(image_dir))

    # get the filename, label and image binary as a dict
    for label in dir_paths:
        for item in os.listdir(dir_paths[label]):
            file_name = os.path.join(dir_paths[label], item)
            if not item.endswith("JPEG") and not item.endswith("jpg"):
                print("{} file is not suffix with JPEG/jpg, skip it.".format(file_name))
                continue

            # get the image data
            image_file = open(file_name, "rb")
            image_bytes = image_file.read()
            image_file.close()
            if not image_bytes:
                print("The image file: {} is invalid.".format(file_name))
                continue
            yield str(file_name), int(label), image_bytes


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))
    return feature


def create_string_feature(values):
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(values, encoding='utf-8')]))
    return feature


def create_bytes_feature(values):
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
    return feature


def imagenet_to_tfrecord():
    writers = []
    for i in range(PARTITION_NUMBER):
        output_file = TFRECORD_FILE + str(i).rjust(2, '0')
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    for file_name, label, image_bytes in get_imagenet_filename_label_pic(IMAGENET_MAP_FILE,
                                                                         IMAGENET_IMAGE_DIR):
        features = collections.OrderedDict()
        features["file_name"] = create_string_feature(file_name)
        features["label"] = create_int_feature(label)
        features["data"] = create_bytes_feature(image_bytes)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

    for writer in writers:
        writer.close()

    print("Write {} total examples".format(total_written))


if __name__ == '__main__':
    imagenet_to_tfrecord()
