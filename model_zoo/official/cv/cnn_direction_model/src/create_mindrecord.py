# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import os
from mindspore.mindrecord import FileWriter
from config import config1 as config

FAIL = 1
SUCCESS = 0

def get_images(image_dir, annot_files):
    """
    Get file paths that are in image_dir, annotation file is used to get the file names.

    Args:
        image_dir(string): images directory.
        annot_files(list(string)) : annotation files.

    Returns:
        status code(int), status of process(string), image ids(list(int)), image paths(dict(int,string))
    """
    print("Process [Get Images] started")
    if not os.path.isdir(image_dir):
        return FAIL, "{} is not a directory. Please check the src/config.py file.".format(image_dir), [], {}
    image_files_dict = {}
    images = []
    img_id = 0
    # create a dictionary of image file paths
    for annot_file in annot_files:
        if not os.path.exists(annot_file):
            return FAIL, "{} was not found.".format(annot_file), [], {}
        lines = open(annot_file, 'r').readlines()
        for line in lines:
            # extract file name
            file_name = line.split('\t')[0]
            image_path = os.path.join(image_dir, file_name)
            if not os.path.isfile(image_path):
                return FAIL, "{} is not a file.".format(image_path), [], {}
            # add path to dictionary
            images.append(img_id)
            image_files_dict[img_id] = image_path
            img_id += 1
    return SUCCESS, "Successfully retrieved {} images.".format(str(len(images))), images, image_files_dict

def write_mindrecord_images(image_ids, image_dict, mindrecord_dir, data_schema, file_num=8):
    writer = FileWriter(os.path.join(mindrecord_dir, config.dataset_name + ".mindrecord"), shard_num=file_num)
    writer.add_schema(data_schema, config.dataset_name)
    len_image_dict = len(image_dict)
    sample_count = 0
    for img_id in image_ids:
        image_path = image_dict[img_id]
        with open(image_path, 'rb') as f:
            img = f.read()
        row = {"image": img}
        sample_count += 1
        writer.write_raw_data([row])
        print("Progress {} / {}".format(str(sample_count), str(len_image_dict)), end='\r')
    writer.commit()

def create_mindrecord():

    annot_files_train = [config.train_annotation_file]
    annot_files_test = [config.test_annotation_file]
    ret_code, ret_message, images_train, image_path_dict_train = get_images(image_dir=config.data_root_train,
                                                                            annot_files=annot_files_train)
    if ret_code != SUCCESS:
        return ret_code, message, "", ""
    ret_code, ret_message, images_test, image_path_dict_test = get_images(image_dir=config.data_root_test,
                                                                          annot_files=annot_files_test)
    if ret_code != SUCCESS:
        return ret_code, ret_message, "", ""
    data_schema = {"image": {"type": "bytes"}}
    train_target = os.path.join(config.mindrecord_dir, "train")
    test_target = os.path.join(config.mindrecord_dir, "test")
    if not os.path.exists(train_target):
        os.mkdir(train_target)
    if not os.path.exists(test_target):
        os.mkdir(test_target)
    print("Creating training mindrecords: ")
    write_mindrecord_images(images_train, image_path_dict_train, train_target, data_schema)
    print("Creating test mindrecords: ")
    write_mindrecord_images(images_test, image_path_dict_test, test_target, data_schema)
    return SUCCESS, "Successful mindrecord creation.", train_target, test_target




if __name__ == "__main__":
    # start creating mindrecords from raw images and annots
    # provide root path to raw data in the config file
    code, message, train_target_dir, test_target_dir = create_mindrecord()
    if code != SUCCESS:
        print("Process done with status code: {}. Error: {}".format(code, message))
    else:
        print("Process done with status: {}. Training and testing data are saved to {} and {} respectively."
              .format(message, train_target_dir, test_target_dir))
