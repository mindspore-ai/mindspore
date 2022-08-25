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
    """自定义数据集类"""
    def __init__(self):
        """自定义初始化操作"""
        self.data = []  # 自定义数据
        # 画圆形
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse(((0, 0), (100, 100)), fill=(255, 0, 0), outline=(255, 0, 0), width=5)
        img.save("./1.jpg")
        with open("./1.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # 画三角形
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.polygon([(50, 50), (150, 50), (100, 150)], fill=(0, 255, 0), outline=(0, 255, 0))
        img.save("./2.jpg")
        with open("./2.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # 画正方形
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(((200, 200), (300, 300)), fill=(0, 0, 255), outline=(0, 0, 255), width=5)
        img.save("./3.jpg")
        with open("./3.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # 画长方形
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(((50, 50), (250, 150)), fill=(0, 255, 255), outline=(0, 255, 255), width=5)
        img.save("./4.jpg")
        with open("./4.jpg", "rb") as f:
            data = f.read()
            self.data.append(data)

        # 自定义标签
        self.label = [1, 2, 3, 4]

    def __getitem__(self, index):
        """自定义随机访问函数"""
        return self.data[index], self.label[index]

    def __len__(self):
        """自定义获取样本数据量函数"""
        return len(self.data)

# 实例化数据集类
dataset_generator = MyDataset()

# 加载数据集
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"])

# 对data数据增强
dataset = dataset.map(operations=vision.Decode(), input_columns="data")
dataset = dataset.map(operations=vision.RandomCrop(size=(250, 250)), input_columns="data")
dataset = dataset.map(operations=vision.Resize(size=(224, 224)), input_columns="data")
dataset = dataset.map(operations=vision.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                  std=[0.229 * 255, 0.224 * 255, 0.225 * 255]),
                      input_columns="data")
dataset = dataset.map(operations=vision.HWC2CHW(), input_columns="data")

# 对label变换类型
dataset = dataset.map(operations=transforms.TypeCast(ms.int32), input_columns="label")

# batch操作
dataset = dataset.batch(batch_size=2)

# 创建迭代器
epochs = 2
ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
for _ in range(epochs):
    for item in ds_iter:
        print("item: {}".format(item), flush=True)
