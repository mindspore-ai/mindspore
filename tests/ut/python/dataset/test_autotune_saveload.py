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
import os
import json
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Border, Inter

MNIST_DATA_DIR = "../data/dataset/testMnistData"
DATA_DIR = "../data/dataset/testPK/data"


def data_pipeline_same(file1, file2):
    assert file1.exists()
    assert file2.exists()
    with file1.open() as f1, file2.open() as f2:
        pipeline1 = json.load(f1)
        pipeline1 = pipeline1["tree"] if "tree" in pipeline1 else pipeline1
        pipeline2 = json.load(f2)
        pipeline2 = pipeline2["tree"] if "tree" in pipeline2 else pipeline2
        return pipeline1 == pipeline2


def validate_jsonfile(filepath):
    try:
        file_exist = os.path.exists(filepath)
        with open(filepath, 'r') as jfile:
            loaded_json = json.load(jfile)
    except IOError:
        return False
    return file_exist and isinstance(loaded_json, dict)


@pytest.mark.forked
class TestAutotuneSaveLoad:
    """
    Test AutoTune Save and Load Configuration Support
    Note: Use pytest fixture tmp_path to create files within this temporary directory,
          which is automatically created for each test and deleted at the end of the test.
    """

    @staticmethod
    def setup_method():
        os.environ['RANK_ID'] = '0'

    @staticmethod
    def teardown_method():
        del os.environ['RANK_ID']

    @staticmethod
    def test_autotune_file_overwrite_warn(tmp_path, capfd):
        """
        Feature: Autotuning
        Description: Test overwriting autofile config file produces a warning message
        Expectation: Pipeline runs successfully and warning message is produced
        """
        original_autotune = ds.config.get_enable_autotune()
        config_path = tmp_path / f"test_autotune_generator_atfinal_{os.environ['RANK_ID']}.json"
        config_path.touch()
        ds.config.set_enable_autotune(True, str(tmp_path / "test_autotune_generator_atfinal"))

        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)

        itr = data1.create_dict_iterator(num_epochs=5)
        for _ in range(5):
            for _ in itr:
                pass
        del itr

        _, err = capfd.readouterr()

        assert f"test_autotune_generator_atfinal_{os.environ['RANK_ID']}.json> already exists. " \
               f"File will be overwritten with the AutoTuned data" in err

        ds.config.set_enable_autotune(original_autotune)

    @staticmethod
    def test_autotune_generator_pipeline(tmp_path):
        """
        Feature: Autotuning
        Description: Test save final config with GeneratorDataset pipeline: Generator -> Shuffle -> Batch
        Expectation: Pipeline runs successfully
        """
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path / "test_autotune_generator_atfinal"))

        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)

        ds.serialize(data1, str(tmp_path / "test_autotune_generator_serialized.json"))

        itr = data1.create_dict_iterator(num_epochs=5)
        for _ in range(5):
            for _ in itr:
                pass
        del itr
        ds.config.set_enable_autotune(original_autotune)

        file = tmp_path / ("test_autotune_generator_atfinal_" + os.environ['RANK_ID'] + ".json")
        assert file.exists()
        validate_jsonfile(file)

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
        ds.config.set_enable_autotune(True, str(tmp_path / at_final_json_filename))

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
    def test_autotune_mnist_pipeline(tmp_path):
        """
        Feature: Autotuning
        Description: Test save final config with Mnist pipeline: Mnist -> Batch -> Map
        Expectation: Pipeline runs successfully
        """
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path / "test_autotune_mnist_pipeline_atfinal"))
        original_seed = ds.config.get_seed()
        ds.config.set_seed(1)

        data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=100)
        one_hot_encode = transforms.OneHot(10)  # num_classes is input argument
        data1 = data1.map(operations=one_hot_encode, input_columns="label")

        data1 = data1.batch(batch_size=10, drop_remainder=True)

        ds.serialize(data1, str(tmp_path / "test_autotune_mnist_pipeline_serialized.json"))

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(original_autotune)

        # Confirm final AutoTune config file pipeline is identical to the serialized file pipeline.
        file1 = tmp_path / ("test_autotune_mnist_pipeline_atfinal_" + os.environ['RANK_ID'] + ".json")
        file2 = tmp_path / "test_autotune_mnist_pipeline_serialized.json"
        assert data_pipeline_same(file1, file2)

        desdata1 = ds.deserialize(json_filepath=str(file1))
        desdata2 = ds.deserialize(json_filepath=str(file2))

        num = 0
        for newdata1, newdata2 in zip(desdata1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                      desdata2.create_dict_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_array_equal(newdata1['image'], newdata2['image'])
            np.testing.assert_array_equal(newdata1['label'], newdata2['label'])
            num += 1
        assert num == 10

        ds.config.set_seed(original_seed)

    @staticmethod
    def test_autotune_imagefolder_pipeline_enum_parms(tmp_path):
        """
        Feature: Autotuning
        Description: Test save final config with ImageFolder pipeline
            that contains op with enumerated types (e.g. Border, Inter).
        Expectation: Pipeline runs successfully
        """
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path / "test_autotune_imagefolder_pipeline_atfinal"))
        original_seed = ds.config.get_seed()
        ds.config.set_seed(1)

        data1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=False, num_samples=5)

        # The following map op uses Python implementation of ops
        data1 = data1.map(operations=[vision.Decode(True),
                                      vision.Resize((250, 300), interpolation=Inter.LINEAR),
                                      vision.RandomRotation((90, 90), expand=True, resample=Inter.BILINEAR,
                                                            center=(50, 50), fill_value=(0, 1, 2))
                                      ],
                          input_columns=["image"])

        ds.serialize(data1, str(tmp_path / "test_autotune_imagefolder_pipeline_serialized.json"))

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(original_autotune)

        # Confirm final AutoTune config file pipeline is identical to the serialized file pipeline.
        file1 = tmp_path / ("test_autotune_imagefolder_pipeline_atfinal_" + os.environ['RANK_ID'] + ".json")
        file2 = tmp_path / "test_autotune_imagefolder_pipeline_serialized.json"
        assert data_pipeline_same(file1, file2)

        desdata1 = ds.deserialize(json_filepath=str(file1))
        desdata2 = ds.deserialize(json_filepath=str(file2))

        num = 0
        for newdata1, newdata2 in zip(desdata1.create_dict_iterator(num_epochs=1, output_numpy=True),
                                      desdata2.create_dict_iterator(num_epochs=1, output_numpy=True)):
            np.testing.assert_array_equal(newdata1['image'], newdata2['image'])
            np.testing.assert_array_equal(newdata1['label'], newdata2['label'])
            num += 1
        assert num == 5

        ds.config.set_seed(original_seed)

    @staticmethod
    def test_autotune_pipeline_pyfunc(tmp_path):
        """
        Feature: Autotuning
        Description: Test Autotune with save final config enabled for pipeline with user-defined Python function.
        Expectation: Pipeline runs successfully. Autotune save final config created for pipelines with UDFs.
        """
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path / "test_autotune_pipeline_pyfunc"))
        original_seed = ds.config.get_seed()
        ds.config.set_seed(55)

        data1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=False, num_samples=5)

        # The following map op uses a user-defined Python function
        data1 = data1.map(operations=[vision.Decode(True),
                                      vision.RandomHorizontalFlip(1.0),
                                      lambda x: x],
                          input_columns=["image"])

        num = 0
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            num += 1
        assert num == 5

        # Confirm that autotune final config file exists.
        atfinal_filename = tmp_path / ("test_autotune_pipeline_pyfunc_" + os.environ['RANK_ID'] + ".json")
        assert atfinal_filename.exists()
        validate_jsonfile(atfinal_filename)

        ds.config.set_enable_autotune(False)

        # Pipeline#2
        ds.config.set_enable_autotune(True, str(tmp_path / "test_autotune_pipeline_pyfunc2"))

        # Execute similar pipeline without user-defined Python function
        data2 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=False, num_samples=6)

        data2 = data2.map(operations=[vision.Decode(True),
                                      vision.RandomVerticalFlip(1.0)],
                          input_columns=["image"])

        num = 0
        for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            num += 1
        assert num == 6

        # Confirm that autotune final config file exists
        atfinal_filename2 = tmp_path / ("test_autotune_pipeline_pyfunc2_" + os.environ['RANK_ID'] + ".json")
        assert atfinal_filename2.exists()
        validate_jsonfile(atfinal_filename2)

        ds.config.set_enable_autotune(original_autotune)
        ds.config.set_seed(original_seed)

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
        dataset = ds.ImageFolderDataset(DATA_DIR, num_samples=8)
        dataset = dataset.map(operations=[vision.Decode()], input_columns="image")
        dataset = dataset.map(operations=[vision.HWC2CHW()], input_columns="image", offload=True)
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
        at_final_json_filename = "test_autotune_save_overwrite_mnist_atfinal"

        # Pipeline#1
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path / at_final_json_filename))

        data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=100)
        one_hot_encode = transforms.OneHot(10)  # num_classes is input argument
        data1 = data1.map(operations=one_hot_encode, input_columns="label")
        data1 = data1.batch(batch_size=10, drop_remainder=True)

        ds.serialize(data1, str(tmp_path / "test_autotune_save_overwrite_mnist_serialized1.json"))

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        # Pipeline#2
        ds.config.set_enable_autotune(True, str(tmp_path / at_final_json_filename))

        data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=200)
        data1 = data1.map(operations=one_hot_encode, input_columns="label")
        data1 = data1.shuffle(40)
        data1 = data1.batch(batch_size=20, drop_remainder=False)

        ds.serialize(data1, str(tmp_path / "test_autotune_save_overwrite_mnist_serialized2.json"))

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        # Confirm 2nd serialized file is identical to final AutoTune config file.
        file1 = tmp_path / ("test_autotune_save_overwrite_mnist_atfinal_" + os.environ['RANK_ID'] + ".json")
        file2 = tmp_path / "test_autotune_save_overwrite_mnist_serialized2.json"
        assert data_pipeline_same(file1, file2)

        # Confirm the serialized files for the 2 different pipelines are different
        file1 = tmp_path / "test_autotune_save_overwrite_mnist_serialized1.json"
        file2 = tmp_path / "test_autotune_save_overwrite_mnist_serialized2.json"
        assert not data_pipeline_same(file1, file2)

        ds.config.set_seed(original_seed)
        ds.config.set_enable_autotune(original_autotune)

    @staticmethod
    def test_autotune_save_overwrite_imagefolder_enum_parms(tmp_path):
        """
        Feature: Autotuning
        Description: Test set_enable_autotune and existing json_filepath is overwritten with dataset pipeline
            that contains op with enumerated types (e.g. Border, Inter).
        Expectation: set_enable_autotune() executes successfully with file-exist warning produced.
            Execution of 2nd pipeline overwrites AutoTune configuration file of 1st pipeline.
        """
        original_seed = ds.config.get_seed()
        ds.config.set_seed(1)
        at_final_json_filename = "test_autotune_save_overwrite_imagefolder_atfinal"

        # Pipeline#1
        original_autotune = ds.config.get_enable_autotune()
        ds.config.set_enable_autotune(True, str(tmp_path / at_final_json_filename))

        data1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=False, num_samples=5)

        # The following map op uses Python implementation of ops
        data1 = data1.map(operations=[vision.Decode(True),
                                      vision.Resize((250, 300), interpolation=Inter.LINEAR),
                                      vision.RandomCrop(size=250, padding=[100, 100, 100, 100],
                                                        padding_mode=Border.EDGE, fill_value=(0, 124, 255)),
                                      vision.RandomAffine(degrees=15, translate=(-0.1, 0.1, 0, 0), scale=(0.9, 1.1),
                                                          resample=Inter.NEAREST)
                                      ],
                          input_columns=["image"])

        ds.serialize(data1, str(tmp_path / "test_autotune_save_overwrite_imagefolder_serialized1.json"))

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        # Pipeline#2
        ds.config.set_enable_autotune(True, str(tmp_path / at_final_json_filename))

        data1 = ds.ImageFolderDataset(DATA_DIR, shuffle=False, decode=False, num_samples=5)

        # The following map op uses Python implementation of ops
        data1 = data1.map(operations=[vision.Decode(True),
                                      vision.RandomRotation((90, 90), expand=True, resample=Inter.BILINEAR,
                                                            center=(50, 50), fill_value=(0, 124, 255)),
                                      vision.RandomPerspective(0.3, 1.0, Inter.LINEAR)
                                      ],
                          input_columns=["image"])

        ds.serialize(data1, str(tmp_path / "test_autotune_save_overwrite_imagefolder_serialized2.json"))

        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

        ds.config.set_enable_autotune(False)

        # Confirm 2nd serialized file is identical to final AutoTune config file.
        file1 = tmp_path / ("test_autotune_save_overwrite_imagefolder_atfinal_" + os.environ['RANK_ID'] + ".json")
        file2 = tmp_path / "test_autotune_save_overwrite_imagefolder_serialized2.json"
        assert data_pipeline_same(file1, file2)

        # Confirm the serialized files for the 2 different pipelines are different
        file1 = tmp_path / "test_autotune_save_overwrite_imagefolder_serialized1.json"
        file2 = tmp_path / "test_autotune_save_overwrite_imagefolder_serialized2.json"
        assert not data_pipeline_same(file1, file2)

        ds.config.set_seed(original_seed)
        ds.config.set_enable_autotune(original_autotune)
