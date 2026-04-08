# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""
Test Map op on windows
"""

import os
import numpy as np

import mindspore.dataset as ds
from mindspore.dataset.transforms import transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter
from mindspore.mindrecord import FileWriter
from tests.mark_utils import arg_mark

PWD = os.path.dirname(__file__)

@arg_mark(plat_marks=['cpu_windows'], level_mark='level1', card_mark='onecard')
def test_map_call_py_transforms_func():
    """
    Feature: Test map with multi py_transforms
    Description: Test map with multi py_transforms
    Expectation: Success
    """
    # create new mindrecord
    file_name = "./test_map_call_py_transforms_func.mindrecord"
    if os.path.exists("{}".format(file_name)):
        os.remove("{}".format(file_name))
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))
    writer = FileWriter(file_name)
    cv_schema_json = {"label": {"type": "int32"},
                      "data": {"type": "bytes"}}
    writer.add_schema(cv_schema_json, "img_schema")
    input_apple_jpg = PWD + "/data/apple.jpg"
    image_file = open(input_apple_jpg, "rb")
    image = image_file.read()
    image_file.close()
    for i in range(100):
        raw_data = [{"data": image, "label": i}]
        writer.write_raw_data(raw_data)
    writer.commit()

    indices = [0, 1, 2, 3, 7, 10, 12, 14, 16, 18, 19, 20, 25, 26, 27, 28, 29, 35]
    sampler = ds.SubsetRandomSampler(indices)

    crop_height = 100
    crop_width = 50
    target_height = 200
    target_width = 200
    scalelb = 0.5
    scaleub = 200.5
    aspectlb = 200.5
    aspectub = 200.5
    targetheight = 100
    targetwidth = 100
    interpolation = Inter.BILINEAR
    maxiter = 100
    transformation_matrix = np.ones([432, 432])
    mean_vector = np.ones([432])
    l1 = []

    dataset = ds.MindDataset(file_name, columns_list=["data", "label"], sampler=sampler, num_parallel_workers=3)
    dataset_num = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        dataset_num += 1
    assert dataset_num == 18

    op_list = [vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                 fill_value=(1, 1, 0), padding_mode=vision.Border.CONSTANT),
               vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                 fill_value=(1, 1, 0), padding_mode=vision.Border.EDGE),
               vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                 fill_value=(1, 1, 0), padding_mode=vision.Border.REFLECT),
               vision.RandomCrop(size=(crop_height, crop_width), padding=(1, 1), pad_if_needed=True,
                                 fill_value=(1, 1, 0), padding_mode=vision.Border.SYMMETRIC),
               vision.RandomHorizontalFlip(),
               vision.RandomVerticalFlip(),
               vision.Grayscale(3),
               vision.RandomGrayscale(0.3),
               vision.RandomPerspective(distortion_scale=0.5, prob=0.1, interpolation=Inter.BICUBIC),
               vision.RandomPerspective(distortion_scale=0.5, prob=0.1, interpolation=Inter.NEAREST),
               vision.RandomPerspective(distortion_scale=0.5, prob=0.1, interpolation=Inter.BILINEAR),
               vision.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
               vision.RandomSharpness((0.1, 1.9)),
               vision.RandomColor((0.1, 1.9)),
               vision.RandomResizedCrop((targetheight, targetwidth), (scalelb, scaleub),
                                        (aspectlb, aspectub), interpolation, maxiter),
               vision.AutoContrast(cutoff=10.0, ignore=[10, 20]),
               vision.Equalize(),
               vision.Invert()
               ]

    operations = transforms.Compose([vision.Decode(to_pil=True),
                                     vision.UniformAugment(transforms=op_list),
                                     vision.Resize((224, 224)),
                                     vision.ToTensor()])

    dataset = dataset.map(operations=operations, input_columns="data", num_parallel_workers=3,
                          python_multiprocessing=True)
    dataset = dataset.shuffle(2)

    def add_one_by_batch_num(batch_info):
        return batch_info.get_batch_num() + 1

    def invert_sign_per_batch_multi_col(col_list, batch_info):
        return ([np.copy(((-1) ** batch_info.get_batch_num()) * arr) for arr in col_list],)

    dataset = dataset.batch(batch_size=add_one_by_batch_num, drop_remainder=True, num_parallel_workers=3,
                            input_columns=["data"], per_batch_map=invert_sign_per_batch_multi_col)

    dataset = dataset.repeat(10)

    for data in dataset.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset_1 = ds.MindDataset(file_name, columns_list=["data", "label"], sampler=sampler, num_parallel_workers=3)
    transform_list = [
        vision.Resize((target_height, target_width)),
        vision.CenterCrop(1),
    ]
    op_list_1 = [
        vision.Decode(to_pil=True),
        vision.Resize((target_height, target_width)),
        vision.CenterCrop(1),
        vision.Pad(padding=(2, 2), padding_mode=vision.Border.CONSTANT),
        vision.Pad(padding=(2, 2), padding_mode=vision.Border.EDGE),
        vision.Pad(padding=(2, 2), padding_mode=vision.Border.REFLECT),
        vision.Pad(padding=(2, 2), padding_mode=vision.Border.SYMMETRIC),
        vision.RandomColorAdjust(brightness=(1.0, 1.0), contrast=(1, 1), saturation=(1, 1), hue=(0, 0)),
        vision.RandomRotation(degrees=(0, 125), resample=Inter.BILINEAR, expand=False, center=(6, 6), fill_value=1),
        vision.RandomRotation(degrees=(0, 125), resample=Inter.NEAREST, expand=False, center=(6, 6), fill_value=1),
        vision.RandomRotation(degrees=(0, 125), resample=Inter.BICUBIC, expand=False, center=(6, 6), fill_value=1),
        transforms.RandomChoice(transform_list),
        transforms.RandomApply(transform_list, prob=0.5),
        transforms.RandomOrder(transform_list),
        vision.Resize(12, interpolation),
        vision.ToTensor(),
        vision.RandomErasing(prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_attempts=10),
        vision.LinearTransformation(transformation_matrix, mean_vector)

    ]
    operations_1 = transforms.Compose(op_list_1)

    dataset_1 = dataset_1.map(operations=operations_1, input_columns="data", num_parallel_workers=8,
                              python_multiprocessing=True)
    dataset_1 = dataset_1.shuffle(2)

    def add_one_by_epoch(batch_info):
        return batch_info.get_epoch_num() + 1

    dataset_1 = dataset_1.padded_batch(batch_size=add_one_by_epoch, drop_remainder=True, num_parallel_workers=3,
                                       pad_info={"label": (None, 2)})

    dataset_1 = dataset_1.repeat(10)
    for data in dataset_1.create_dict_iterator(output_numpy=True):
        l1.append(data['data'])
    l1.clear()

    dataset_2 = ds.MindDataset(file_name, sampler=sampler, num_parallel_workers=3)
    op_list_2 = [
        vision.Decode(to_pil=True),
        vision.FiveCrop(size=(2, 2)),
        vision.TenCrop(size=(2, 2)),
        lambda images: np.stack([vision.ToTensor()(image) for image in images]),
        vision.ToType(np.float32),
        vision.ToPIL()
    ]
    operations_2 = transforms.Compose(op_list_2)
    dataset_2.map(operations=operations_2, input_columns=["image"], num_parallel_workers=3)
    for data in dataset_2.create_dict_iterator(output_numpy=True):
        l1.append(data["data"])
    l1.clear()

    # remove mindrecord
    if os.path.exists("{}".format(file_name)):
        os.remove("{}".format(file_name))
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))

if __name__ == '__main__':
    test_map_call_py_transforms_func()
