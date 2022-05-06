# Copyright 2022 Huawei Technologies Co., Ltd
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
Test Map op in Dataset
"""
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as text
from mindspore.dataset.transforms import transforms
import mindspore.dataset.vision.transforms as vision

DATA_DIR = "../data/dataset/testPK/data"


def test_map_c_transform_exception():
    """
    Feature: test c error op def
    Description: op defined like vision.HWC2CHW
    Expectation: success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR, num_parallel_workers=1, shuffle=True)

    train_image_size = 224
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    random_crop_decode_resize_op = vision.RandomCropDecodeResize(train_image_size,
                                                                 scale=(0.08, 1.0),
                                                                 ratio=(0.75, 1.333))
    random_horizontal_flip_op = vision.RandomHorizontalFlip(prob=0.5)
    normalize_op = vision.Normalize(mean=mean, std=std)
    hwc2chw_op = vision.HWC2CHW  # exception

    data_set = data_set.map(operations=random_crop_decode_resize_op, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=random_horizontal_flip_op, input_columns="image", num_parallel_workers=1)
    data_set = data_set.map(operations=normalize_op, input_columns="image", num_parallel_workers=1)
    with pytest.raises(ValueError) as info:
        data_set = data_set.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=1)
    assert "Parameter operations's element of method map should be a " in str(info.value)

    # compose exception
    with pytest.raises(ValueError) as info:
        transforms.Compose([
            vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip,
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()])
    assert " should be a " in str(info.value)

    # randomapply exception
    with pytest.raises(ValueError) as info:
        transforms.RandomApply([
            vision.RandomCropDecodeResize,
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()])
    assert " should be a " in str(info.value)

    # randomchoice exception
    with pytest.raises(ValueError) as info:
        transforms.RandomChoice([
            vision.RandomCropDecodeResize(train_image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize,
            vision.HWC2CHW()])
    assert " should be a " in str(info.value)


def test_map_py_transform_exception():
    """
    Feature: test python error op def
    Description: op defined like vision.RandomHorizontalFlip
    Expectation: success
    """
    data_set = ds.ImageFolderDataset(DATA_DIR, num_parallel_workers=1, shuffle=True)

    # define map operations
    decode_op = vision.Decode(to_pil=True)
    random_horizontal_flip_op = vision.RandomHorizontalFlip  # exception
    to_tensor_op = vision.ToTensor()
    trans = [decode_op, random_horizontal_flip_op, to_tensor_op]

    with pytest.raises(ValueError) as info:
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=1)
    assert "Parameter operations's element of method map should be a " in str(info.value)

    # compose exception
    with pytest.raises(ValueError) as info:
        transforms.Compose([
            vision.Decode,
            vision.RandomHorizontalFlip(),
            vision.ToTensor()])
    assert " should be a " in str(info.value)

    # randomapply exception
    with pytest.raises(ValueError) as info:
        transforms.RandomApply([
            vision.Decode(to_pil=True),
            vision.RandomHorizontalFlip,
            vision.ToTensor()])
    assert " should be a " in str(info.value)

    # randomchoice exception
    with pytest.raises(ValueError) as info:
        transforms.RandomChoice([
            vision.Decode(to_pil=True),
            vision.RandomHorizontalFlip(),
            vision.ToTensor])
    assert " should be a " in str(info.value)


def test_map_text_and_data_transforms():
    """
    Feature: Map op
    Description: Test Map op with both Text Transforms and Data Transforms
    Expectation: Dataset pipeline runs successfully and results are verified
    """
    data = ds.TextFileDataset("../data/dataset/testVocab/words.txt", shuffle=False)

    vocab = text.Vocab.from_dataset(data, "text", freq_range=None, top_k=None,
                                    special_tokens=["<pad>", "<unk>"],
                                    special_first=True)

    padend_op = transforms.PadEnd([100], pad_value=vocab.tokens_to_ids('<pad>'))
    lookup_op = text.Lookup(vocab, "<unk>")

    # Use both Text Lookup op and Data Transforms PadEnd op in operations list for Map
    data = data.map(operations=[lookup_op, padend_op], input_columns=["text"])
    res = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res.append(d["text"].item())
    assert res == [4, 5, 3, 6, 7, 2], res


if __name__ == '__main__':
    test_map_c_transform_exception()
    test_map_py_transform_exception()
    test_map_text_and_data_transforms()
