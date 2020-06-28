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
import numpy as np
from mindspore.mindrecord import FileWriter

CUB_200_2011_DIR = "data/CUB_200_2011"
SEGMENTATION_DIR = "data/segmentations"

MINDRECORD_FILE_NAME = "output/CUB_200_2011.mindrecord"

def get_data_as_dict():
    """get data from dataset"""
    # id : filename
    id_and_filename = {}
    images_txt = open(os.path.join(CUB_200_2011_DIR, "images.txt"))
    for line in images_txt:
        # images.txt, get id and filename
        single_images_txt = line.split(" ")
        id_and_filename[int(single_images_txt[0])] = os.path.join(os.path.join(CUB_200_2011_DIR, "images"),
                                                                  single_images_txt[1].replace("\n", ""))
    images_txt.close()

    # id : bounding_box
    id_and_bbox = {}
    bounding_boxes_txt = open(os.path.join(CUB_200_2011_DIR, "bounding_boxes.txt"))
    for line in bounding_boxes_txt:
        # bounding_boxes.txt, get id and bounding_box
        single_bounding_boxes_txt = line.split(" ")
        id_and_bbox[int(single_bounding_boxes_txt[0])] = [float(single_bounding_boxes_txt[1]),
                                                          float(single_bounding_boxes_txt[2]),
                                                          float(single_bounding_boxes_txt[3]),
                                                          float(single_bounding_boxes_txt[4])]
    bounding_boxes_txt.close()

    # id : label
    id_and_label = {}
    image_class_labels_txt = open(os.path.join(CUB_200_2011_DIR, "image_class_labels.txt"))
    for line in image_class_labels_txt:
        # image_class_labels.txt, get id and label
        single_image_class_labels_txt = line.split(" ")
        id_and_label[int(single_image_class_labels_txt[0])] = int(single_image_class_labels_txt[1])
    image_class_labels_txt.close()

    # id : segmentation filename
    id_and_segmentation_file_name = {}
    for item in id_and_filename:
        segmentation_filename = id_and_filename[item]
        segmentation_filename = segmentation_filename.replace(os.path.join(CUB_200_2011_DIR, "images"),
                                                              SEGMENTATION_DIR)
        segmentation_filename = segmentation_filename.replace(".jpg", ".png")
        id_and_segmentation_file_name[item] = segmentation_filename

    # label: class
    label_and_class = {}
    classes_txt = open(os.path.join(CUB_200_2011_DIR, "classes.txt"))
    for line in classes_txt:
        # classes.txt, get label and class
        single_classes_txt = line.split(" ")
        label_and_class[int(single_classes_txt[0])] = str(single_classes_txt[1]).replace("\n", "")
    classes_txt.close()

    assert len(id_and_filename) == len(id_and_bbox)
    assert len(id_and_filename) == len(id_and_label)
    assert len(id_and_filename) == len(id_and_segmentation_file_name)

    print(">> sample id: {}, filename: {}, bbox: {}, label: {}, seg_filename: {}, class: {}"
          .format(1, id_and_filename[1], id_and_bbox[1], id_and_label[1], id_and_segmentation_file_name[1],
                  label_and_class[id_and_label[1]]))

    for item in id_and_filename:
        data = {}
        data["bbox"] = np.asarray(id_and_bbox[item], dtype=np.float32)  # [60.0, 27.0, 325.0, 304.0]

        image_file = open(id_and_filename[item], "rb")
        image_bytes = image_file.read()
        image_file.close()
        data["image"] = image_bytes

        image_filename = id_and_filename[item].split("/")
        data["image_filename"] = image_filename[-1]  # Black_Footed_Albatross_0046_18.jpg

        data["label"] = id_and_label[item]  # 1-200
        data["label_name"] = label_and_class[id_and_label[item]]  # 177.Prothonotary_Warbler

        segmentation_file = open(id_and_segmentation_file_name[item], "rb")
        segmentation_bytes = segmentation_file.read()
        segmentation_file.close()
        data["segmentation_mask"] = segmentation_bytes

        yield data

def gen_mindrecord():
    """gen mindreocrd according exactly schema"""
    fw = FileWriter(MINDRECORD_FILE_NAME)

    schema = {"bbox": {"type": "float32", "shape": [-1]},
              "image": {"type": "bytes"},
              "image_filename": {"type": "string"},
              "label": {"type": "int32"},
              "label_name": {"type": "string"},
              "segmentation_mask": {"type": "bytes"}}
    fw.add_schema(schema, "CUB 200 2011 dataset")

    get_data_iter = get_data_as_dict()

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
    # generate mindrecord
    print(">> begin generate mindrecord")
    gen_mindrecord()

if __name__ == "__main__":
    main()
