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
Test Dataset AutoTune Configuration Support
"""
import pytest
import mindspore.dataset as ds


@pytest.mark.forked
class TestAutotuneConfig:
    @staticmethod
    def test_autotune_config_basic():
        """
        Feature: Autotuning
        Description: Test basic config of AutoTune
        Expectation: Config can be set successfully
        """
        autotune_state = ds.config.get_enable_autotune()
        assert autotune_state is False

        ds.config.set_enable_autotune(False)
        autotune_state = ds.config.get_enable_autotune()
        assert autotune_state is False

        with pytest.raises(TypeError):
            ds.config.set_enable_autotune(1)

        autotune_interval = ds.config.get_autotune_interval()
        assert autotune_interval == 0

        ds.config.set_autotune_interval(200)
        autotune_interval = ds.config.get_autotune_interval()
        assert autotune_interval == 200

        with pytest.raises(TypeError):
            ds.config.set_autotune_interval(20.012)

        with pytest.raises(ValueError):
            ds.config.set_autotune_interval(-999)

    @staticmethod
    def test_autotune_config_filepath_invalid():
        """
        Feature: Autotuning
        Description: Test set_enable_autotune() with invalid json_filepath
        Expectation: Invalid input is detected
        """
        with pytest.raises(TypeError):
            ds.config.set_enable_autotune(True, 123)

        with pytest.raises(TypeError):
            ds.config.set_enable_autotune(True, 0)

        with pytest.raises(TypeError):
            ds.config.set_enable_autotune(True, True)

        with pytest.raises(TypeError):
            ds.config.set_enable_autotune(False, 1.1)

        with pytest.raises(RuntimeError) as error_info:
            ds.config.set_enable_autotune(True, "")
            assert "cannot be the empty string" in str(error_info.value)

        with pytest.raises(RuntimeError) as error_info:
            ds.config.set_enable_autotune(True, "/tmp")
            assert "is a directory" in str(error_info.value)

        with pytest.raises(RuntimeError) as error_info:
            ds.config.set_enable_autotune(True, ".")
            assert "is a directory" in str(error_info.value)

        with pytest.raises(RuntimeError) as error_info:
            ds.config.set_enable_autotune(True, "/JUNKPATH/at_out.json")
            assert "Directory" in str(error_info.value)
            assert "does not exist" in str(error_info.value)


    @staticmethod
    def test_autotune_config_filepath_success():
        """
        Feature: Autotuning
        Description: Test set_enable_autotune() with valid filepath input
        Expectation: set_enable_autotune() executes successfully
        """
        # Note: No problem to have sequential calls to set_enable_autotune()
        ds.config.set_enable_autotune(True, "file1.json")
        ds.config.set_enable_autotune(True, "file1.json")
        ds.config.set_enable_autotune(True, "file2.json")

        # Note: It is permissible to not have preferred '.json' extension for json_filepath
        ds.config.set_enable_autotune(True, "at_out.JSON")
        ds.config.set_enable_autotune(True, "/tmp/at_out.txt")
        ds.config.set_enable_autotune(True, "at_out")

        # Note: When enable is false, the json_filepath parameter is ignored
        ds.config.set_enable_autotune(False, "/NONEXISTDIR/junk.json")
        ds.config.set_enable_autotune(False, "")

        ds.config.set_enable_autotune(False, None)
