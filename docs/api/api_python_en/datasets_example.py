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
This example mainly illustrates the usage of mindspore data processing pipeline.
"""

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from PIL import Image, ImageDraw


class MyDataset:
    """User-defined Loader"""
    def __init__(self):
        self.data = []  # user-defined data
        # draw ellipse
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        img.save("./1.jpg")
        with open("./1.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # draw triangle
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        img.save("./2.jpg")
        with open("./2.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # draw square
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(((200, 200), (300, 300)), fill=(0, 0, 255), outline=(0, 0, 255), width=5)
        img.save("./3.jpg")
        with open("./3.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # draw rectangle
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(((50, 50), (250, 150)), fill=(0, 255, 255), outline=(0, 255, 255), width=5)
        img.save("./4.jpg")
        with open("./4.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # label
        self.label = [1, 2, 3, 4]

    def __getitem__(self, index):
        """get item"""
        return self.data[index], self.label[index]

    def __len__(self):
        """dataset length"""
        return len(self.data)

# initialize dataset class
dataset_generator = MyDataset()

# load dataset by GeneratorDataset
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"])

# transform the data
dataset = dataset.map(operations=vision.Decode(), input_columns="data")
dataset = dataset.map(operations=vision.RandomCrop(size=(250, 250)), input_columns="data")
dataset = dataset.map(operations=vision.Resize(size=(224, 224)), input_columns="data")
dataset = dataset.map(operations=vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                  std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
                      input_columns="data")
dataset = dataset.map(operations=vision.HWC2CHW(), input_columns="data")

# transform the label
dataset = dataset.map(operations=transforms.TypeCast(ms.int32), input_columns="label")

# batch
dataset = dataset.batch(batch_size=2)

# create iterator
epochs = 2
ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
for _ in range(epochs):
    for item in ds_iter:
        print("item: {}".format(item), flush=True)
