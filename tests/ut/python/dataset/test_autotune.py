# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
Testing Autotune support in DE
"""
import os
import sys
import numpy as np
import pytest
import mindspore._c_dataengine as cde
import mindspore.dataset as ds

def err_out_log(out, err, log=False):
    if log:
        sys.stdout.write(out)
        sys.stderr.write(err)


def test_disable_autotune_in_independent_mode():
    """
    Feature: Disable autotune in independent mode.
    Description: Test autotune is disable in independent mode.
    Expectation: Exception raised to notify feature not supported.
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"
    with pytest.raises(RuntimeError) as err:
        ds.config.set_enable_autotune(True)
    assert "Dataset AutoTune is not supported in Dataset Independent mode" in str(err.value)
    os.environ["MS_INDEPENDENT_DATASET"] = "False"
    ds.config.set_enable_autotune(False)


class TestAutotuneWithProfiler:
    @staticmethod
    def test_autotune_after_profiler_with_1_pipeline(capfd):
        """
        Feature: Autotuning with Profiler
        Description: Test Autotune enabled together with MD Profiler with a single pipeline
        Expectation: Enable MD Profiler and print appropriate warning logs when trying to enable Autotune
        """
        md_profiler = cde.GlobalContext.profiling_manager()
        md_profiler.init()
        md_profiler.start()
        ds.config.set_enable_autotune(True)
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        _ = data1.create_dict_iterator(num_epochs=5)

        out, err = capfd.readouterr()
        assert "Cannot enable AutoTune for the current data pipeline as Dataset Profiling is already enabled for the " \
               "current data pipeline." in err
        # Change False to True in the following call to see complete stdout and stderr output in pytest summary output
        err_out_log(out, err, False)


        md_profiler.stop()
        ds.config.set_enable_autotune(False)

    @staticmethod
    def test_autotune_after_profiler_with_2_pipeline(capfd):
        """
        Feature: Autotuning with Profiler
        Description: Test Autotune enabled together with MD Profiler with two pipelines
        Expectation: Enable MD Profiler for first tree and print appropriate warning log when trying to
        enable Autotune for the first tree. Print appropriate warning logs when trying to enable both MD Profiler
        and Autotune for second tree.
        """
        md_profiler = cde.GlobalContext.profiling_manager()
        md_profiler.init()
        md_profiler.start()
        ds.config.set_enable_autotune(True)
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        _ = data1.create_dict_iterator(num_epochs=5)

        out, err = capfd.readouterr()
        assert "Cannot enable AutoTune for the current data pipeline as Dataset Profiling is already enabled for the " \
               "current data pipeline." in err
        # Change False to True in the following call to see complete stdout and stderr output in pytest summary output
        err_out_log(out, err, False)

        _ = data1.create_dict_iterator(num_epochs=5)

        out, err = capfd.readouterr()
        assert "Dataset Profiling is already enabled for a different data pipeline." in err
        assert "Cannot enable AutoTune for the current data pipeline as Dataset Profiling is enabled for another data" \
               " pipeline." in err
        # Change False to True in the following call to see complete stdout and stderr output in pytest summary output
        err_out_log(out, err, False)

        md_profiler.stop()
        ds.config.set_enable_autotune(False)

    @staticmethod
    def test_autotune_with_2_pipeline(capfd):
        """
        Feature: Autotuning
        Description: Test Autotune two pipelines
        Expectation: Enable MD Profiler and print appropriate warning logs
        when trying to enable Autotune for second tree.
        """
        ds.config.set_enable_autotune(True)
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        _ = data1.create_dict_iterator(num_epochs=5)
        _ = data1.create_dict_iterator(num_epochs=5)

        out, err = capfd.readouterr()
        assert "Cannot enable AutoTune for the current data pipeline as it is already enabled for another data " \
               "pipeline." in err
        # Change False to True in the following call to see complete stdout and stderr output in pytest summary output
        err_out_log(out, err, False)

        ds.config.set_enable_autotune(False)

    @staticmethod
    def test_delayed_autotune_with_2_pipeline(tmp_path, capfd):
        """
        Feature: Autotuning
        Description: Test delayed Autotune with two pipelines
        Expectation: Enable MD Profiler for second tree and no warnings logs should be printed.
        """
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        _ = data1.create_dict_iterator(num_epochs=5)

        ds.config.set_enable_autotune(True, str(tmp_path / "file.json"))
        _ = data1.create_dict_iterator(num_epochs=5)
        ds.config.set_enable_autotune(False)

        out, err = capfd.readouterr()
        assert err == ''
        # Change False to True in the following call to see complete stdout and stderr output in pytest summary output
        err_out_log(out, err, False)

    @staticmethod
    def test_delayed_start_autotune_with_3_pipeline(tmp_path, capfd):
        """
        Feature: Autotuning
        Description: Test delayed Autotune and early stop with three pipelines
        Expectation: Enable MD Profiler for second tree and no warnings logs should be printed.
        """
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        _ = data1.create_dict_iterator(num_epochs=5)

        ds.config.set_enable_autotune(True, str(tmp_path / "file.json"))
        _ = data1.create_dict_iterator(num_epochs=5)
        ds.config.set_enable_autotune(False)

        _ = data1.create_dict_iterator(num_epochs=5)

        out, err = capfd.readouterr()
        assert err == ''
        # Change False to True in the following call to see complete stdout and stderr output in pytest summary output
        err_out_log(out, err, False)

    @staticmethod
    def test_autotune_before_profiler():
        """
        Feature: Autotuning with Profiler
        Description: Test Autotune with Profiler when Profiler is Initialized after autotune
        Expectation: Initialization of Profiler should throw an error.
        """
        # enable AT for 1st tree
        # profiler init should throw an error
        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)
        ds.config.set_enable_autotune(True)
        _ = data1.create_dict_iterator(num_epochs=5)
        ds.config.set_enable_autotune(False)

        md_profiler = cde.GlobalContext.profiling_manager()
        with pytest.raises(RuntimeError) as excinfo:
            md_profiler.init()

        assert "Stop MD Autotune before initializing the MD Profiler." in str(excinfo.value)

    @staticmethod
    def test_autotune_simple_pipeline():
        """
        Feature: Autotuning
        Description: Test simple pipeline of autotune - Generator -> Shuffle -> Batch
        Expectation: Pipeline runs successfully
        """
        ds.config.set_enable_autotune(True)

        source = [(np.array([x]),) for x in range(1024)]
        data1 = ds.GeneratorDataset(source, ["data"])
        data1 = data1.shuffle(64)
        data1 = data1.batch(32)

        itr = data1.create_dict_iterator(num_epochs=5)
        for _ in range(5):
            for _ in itr:
                pass

        ds.config.set_enable_autotune(False)
