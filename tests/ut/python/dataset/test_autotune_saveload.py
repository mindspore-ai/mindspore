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
Test Dataset AutoTune's Save and Load Configuration support
"""
import filecmp

import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.vision.c_transforms as c_vision

MNIST_DATA_DIR = "../data/dataset/testMnistData"
DATA_DIR = "../data/dataset/testPK/data"


@pytest.mark.forked
class TestAutotuneSaveLoad:
    # Note: Use pytest fixture tmp_path to create files within this temporary directory,
    # which is automatically created for each test and deleted at the end of the test.

    @staticmethod
    def test_autotune_generator_pipeline(tmp_path):
        """
        Feature: Autotuning
        Description: Test save final config with GeneratorDataset pipeline: Generator -> Shuffle -> Batch
        Expectation: pipeline runs successfully
        """
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path) + "test_autotune_generator_atfinal.json")

        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)

        ds.serialize(data1, str(tmp_path) + "test_autotune_generator_serialized.json")

        itr = data1.create_dict_iterator(num_epochs=5)
        for _ in range(5):
            for _ in itr:
                pass

        ds.config.set_enable_autotune(original_autotune)

    @staticmethod
    def test_autotune_mnist_pipeline(tmp_path):
        """
        Feature: Autotuning
        Description: Test save final config with Mnist pipeline: Mnist -> Batch -> Map
        Expectation: pipeline runs successfully
        """
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path) + "test_autotune_mnist_pipeline_atfinal.json")
        original_seed = ds.config.get_seed()
        ds.config.set_seed(1)

        data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=100)
        one_hot_encode = c_transforms.OneHot(10)  # num_classes is input argument
        data1 = data1.map(operations=one_hot_encode, input_columns="label")

        data1 = data1.batch(batch_size=10, drop_remainder=True)

        ds.serialize(data1, str(tmp_path) + "test_autotune_mnist_pipeline_serialized.json")

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(original_autotune)

        # Confirm final AutoTune config file is identical to the serialized file.
        assert filecmp.cmp(str(tmp_path) + "test_autotune_mnist_pipeline_atfinal.json",
                           str(tmp_path) + "test_autotune_mnist_pipeline_serialized.json")

        desdata1 = ds.deserialize(json_filepath=str(tmp_path) + "test_autotune_mnist_pipeline_atfinal.json")
        desdata2 = ds.deserialize(json_filepath=str(tmp_path) + "test_autotune_mnist_pipeline_serialized.json")

        num = 0
        for newdata1, newdata2 in zip(desdata1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                      desdata2.create_dict_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_array_equal(newdata1['image'], newdata2['image'])
            np.testing.assert_array_equal(newdata1['label'], newdata2['label'])
            num += 1
        assert num == 10

        ds.config.set_seed(original_seed)

    @staticmethod
    def test_autotune_save_overwrite_generator(tmp_path):
        """
        Feature: Autotuning
        Description: Test set_enable_autotune and existing json_filepath is overwritten
        Expectation: set_enable_autotune() executes successfully with file-exist warning produced.
            Execution of 2nd pipeline overwrites AutoTune configuration file of 1st pipeline.
        """
        source = [(np.array([x]),) for x in range(1024)]

        at_final_json_filename = "test_autotune_save_overwrite_generator_atfinal.json"
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path) + at_final_json_filename)

        data1 = ds.GeneratorDataset(source, ["data"])

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        ds.config.set_enable_autotune(True, str(tmp_path) + at_final_json_filename)

        data2 = ds.GeneratorDataset(source, ["data"])
        data2 = data2.shuffle(64)

        for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(original_autotune)

    @staticmethod
    def test_autotune_save_overwrite_mnist(tmp_path):
        """
        Feature: Autotuning
        Description: Test set_enable_autotune and existing json_filepath is overwritten
        Expectation: set_enable_autotune() executes successfully with file-exist warning produced.
            Execution of 2nd pipeline overwrites AutoTune configuration file of 1st pipeline.
        """
        original_seed = ds.config.get_seed()
        ds.config.set_seed(1)
        at_final_json_filename = "test_autotune_save_overwrite_mnist_atfinal.json"

        # Pipeline#1
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path) + at_final_json_filename)

        data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=100)
        one_hot_encode = c_transforms.OneHot(10)  # num_classes is input argument
        data1 = data1.map(operations=one_hot_encode, input_columns="label")
        data1 = data1.batch(batch_size=10, drop_remainder=True)

        ds.serialize(data1, str(tmp_path) + "test_autotune_save_overwrite_mnist_serialized1.json")

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        # Pipeline#2
        ds.config.set_enable_autotune(True, str(tmp_path) + at_final_json_filename)

        data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=200)
        data1 = data1.map(operations=one_hot_encode, input_columns="label")
        data1 = data1.shuffle(40)
        data1 = data1.batch(batch_size=20, drop_remainder=False)

        ds.serialize(data1, str(tmp_path) + "test_autotune_save_overwrite_mnist_serialized2.json")

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        # Confirm 2nd serialized file is identical to final AutoTune config file.
        assert filecmp.cmp(str(tmp_path) + "test_autotune_save_overwrite_mnist_atfinal.json",
                           str(tmp_path) + "test_autotune_save_overwrite_mnist_serialized2.json")

        # Confirm the serialized files for the 2 different pipelines are different
        assert not filecmp.cmp(str(tmp_path) + "test_autotune_save_overwrite_mnist_serialized1.json",
                               str(tmp_path) + "test_autotune_save_overwrite_mnist_serialized2.json")

        ds.config.set_seed(original_seed)
        ds.config.set_enable_autotune(original_autotune)

    @staticmethod
    def test_autotune_warning_with_offload(tmp_path, capfd):
        """
        Feature: Autotuning
        Description: Test autotune config saving with offload=True
        Expectation: Autotune should not write the config file and print a log message
        """
        original_seed = ds.config.get_seed()
        ds.config.set_seed(1)
        at_final_json_filename = "test_autotune_warning_with_offload_config.json"
        config_path = tmp_path / at_final_json_filename
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(config_path))

        # Dataset with offload activated.
        dataset = ds.ImageFolderDataset(DATA_DIR)
        dataset = dataset.map(operations=[c_vision.Decode()], input_columns="image")
        dataset = dataset.map(operations=[c_vision.HWC2CHW()], input_columns="image", offload=True)
        dataset = dataset.batch(8, drop_remainder=True)

        for _ in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            pass

        _, err = capfd.readouterr()

        assert "Some nodes have been offloaded. AutoTune is unable to write the autotune configuration to disk. " \
               "Disable offload to prevent this from happening." in err

        with pytest.raises(FileNotFoundError):
            with open(config_path) as _:
                pass

        ds.config.set_enable_autotune(original_autotune)
        ds.config.set_seed(original_seed)
