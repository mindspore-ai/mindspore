# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import dataset as ds
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import transforms as CV
from mindspore.dataset.transforms import transforms as C


class FakeDataset:

    @classmethod
    def create_fake_cv_dataset(
            cls,
            batch_size=32,
            repeat_size=1,
            num_parallel_workers=1,
            num_images=1000,
            image_size=(32, 32, 1),
            num_classes=10,
    ):
        """
        Create a fake dataset for CV models.

        Args:
            batch_size (int): The number of samples per batch.
            repeat_size (int): The number of times the dataset is repeated.
            num_parallel_workers (int): The number of workers to use for data processing in parallel.
            num_images (int): The total number of fake images to generate.
            image_size (tuple): The size of each image (height, width, channel).
            num_classes (int): The number of classes for the dataset's labels.

        Returns:
            Dataset: A MindSpore Dataset object containing the fake dataset.

        Examples:
            >>> from mindspore import nn
            >>> from mindspore.train import Model
            >>> from mindspore.train import Accuracy
            >>> from mindspore.nn.optim import Momentum
            >>> from .model_zoo import LeNet5
            >>> # create fake dataset
            >>> cv_dataset = FakeDataset.create_fake_cv_dataset()
            >>> model = LeNet5()
            >>> # train define
            >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
            >>> optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = Model(model, loss_fn=loss, optimizer=optim, metrics={'acc': Accuracy()})
            >>> model.train(3, cv_dataset)
        """
        # define dataset
        fake_dataset = ds.FakeImageDataset(
            num_images=num_images,
            image_size=image_size,
            num_classes=num_classes,
            num_samples=batch_size * 10
        )

        resize_height, resize_width = 32, 32
        rescale = 1.0 / 255.0
        rescale_nml = 1 / 0.3081
        shift_nml = -1 * 0.1307 / 0.3081

        # define map operations
        resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
        rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
        rescale_op = CV.Rescale(rescale, shift=0.0)
        hwc2chw_op = CV.HWC2CHW()
        type_cast_op = C.TypeCast(mstype.int32)

        # apply map operations on images
        fake_dataset = fake_dataset.map(operations=type_cast_op, input_columns="label",
                                        num_parallel_workers=num_parallel_workers)
        fake_dataset = fake_dataset.map(operations=resize_op, input_columns="image",
                                        num_parallel_workers=num_parallel_workers)
        fake_dataset = fake_dataset.map(operations=rescale_op, input_columns="image",
                                        num_parallel_workers=num_parallel_workers)
        fake_dataset = fake_dataset.map(operations=rescale_nml_op, input_columns="image",
                                        num_parallel_workers=num_parallel_workers)
        fake_dataset = fake_dataset.map(operations=hwc2chw_op, input_columns="image",
                                        num_parallel_workers=num_parallel_workers)

        # apply DatasetOps
        fake_dataset = fake_dataset.batch(batch_size, drop_remainder=True)
        fake_dataset = fake_dataset.repeat(repeat_size)

        return fake_dataset

    @classmethod
    def create_fake_nlp_dataset(
            cls,
            seq_len: int = 10,
            batch_size: int = 32,
            d_model: int = 32,
            tgt_len: int = 20,
            num_samples: int = 100,
            num_parallel_workers=1
    ):
        """
        Create a fake dataset for NLP models.

        Args:
            seq_len(int): Sequence length.
            batch_size(int): Size of each batch.
            d_model(int): Dimensionality of the model.
            tgt_len(int): Length of the target sequence.
            num_samples(int): Number of samples in the dataset.
            num_parallel_workers(int): Number of workers to read the data in parallel.

        Returns:
            mindspore.dataset.RandomDataset: A randomly generated dataset according to the specified parameters.

        Examples:
            >>> from .model_zoo import TinyTransformer
            >>> nlp_dataset = FakeDataset.create_fake_nlp_dataset()
            >>> model = TinyTransformer()
            >>> # train loop
            >>> for src, tgt in nlp_dataset:
            >>>     out = model(src, tgt)
        """
        schema = ds.Schema()
        schema.add_column('src', de_type=mstype.float32, shape=[seq_len, batch_size, d_model])
        schema.add_column('tgt', de_type=mstype.float32, shape=[tgt_len, batch_size, d_model])

        # apply dataset operations
        ds.config.set_num_parallel_workers(num_parallel_workers)
        nlp_dataset = ds.RandomDataset(
            schema=schema,
            total_rows=num_samples,
            num_parallel_workers=num_parallel_workers
        )

        return nlp_dataset
