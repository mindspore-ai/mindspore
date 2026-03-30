# Copyright 2026 Huawei Technologies Co., Ltd
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
"""test AdaptTrainOptimizer function."""
#pylint: disable=W0212
import os
import json
import tempfile
from unittest.mock import patch
import pytest
from mindspore.train.callback import _train_fault_tolerance
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig


def clear_env():
    """Clear environment variables."""
    if "MS_ENABLE_TFT" in os.environ:
        del os.environ["MS_ENABLE_TFT"]


def test_ato_basic_and_parameter_validation():
    """
    Feature: ATO basic functionality and parameter validation
    Description: Test ATO basic functionality with valid configuration and parameter validation with invalid inputs
    Expectation: For valid configuration, ATO function enabled successfully; 
                 for invalid inputs, corresponding ValueError exceptions are thrown
    """
    # Test case 1: Normal startup configuration
    os.environ["MS_ENABLE_TFT"] = "ATO:1,ATO_THRESHOLD:10"
    tft = _train_fault_tolerance.TrainFaultTolerance()
    ato_path, ato_threshold = tft._get_ato_env_args()

    assert ato_path == "./ato_info.json"
    assert ato_threshold == 10
    clear_env()

    # Test case 2: ATO with other TFT modes conflict
    os.environ["MS_ENABLE_TFT"] = "ATO:1,TTP:1,ATO_THRESHOLD:10"
    with pytest.raises(ValueError):
        _train_fault_tolerance.TrainFaultTolerance()
    clear_env()

    # Test case 3: ATO threshold as non-numeric value
    os.environ["MS_ENABLE_TFT"] = "ATO:1,ATO_THRESHOLD:abc"
    with pytest.raises(ValueError):
        _train_fault_tolerance.TrainFaultTolerance()
    clear_env()

    # Test case 4: ATO threshold as negative value
    os.environ["MS_ENABLE_TFT"] = "ATO:1,ATO_THRESHOLD:-1"
    with pytest.raises(ValueError):
        _train_fault_tolerance.TrainFaultTolerance()
    clear_env()

    # Test case 5: ATO_PATH as non-json file
    os.environ["MS_ENABLE_TFT"] = "ATO:1,ATO_THRESHOLD:10,ATO_PATH:./test_ato_info.txt"
    with pytest.raises(ValueError):
        _train_fault_tolerance.TrainFaultTolerance()
    clear_env()


def test_ato_adaptive_training_analysis(capsys, caplog):
    """
    Feature: ATO adaptive training optimization analysis
    Description: Test ATO adaptive training optimization analysis under different fault count scenarios
    Expectation: When fault count doesn't reach threshold, no analysis output; 
                 when fault count reaches threshold, calculate analysis and output report
    """
    # Test 1: Fault count below threshold
    os.environ["MS_ENABLE_TFT"] = "ATO:1,ATO_THRESHOLD:10,ATO_PATH:./ato_info_test1.json"
    config = CheckpointConfig(save_checkpoint_steps=100)
    ckpt_cb = ModelCheckpoint(prefix="test", config=config)
    tft_cb = _train_fault_tolerance.TrainFaultTolerance()
    tft_cb.ato._failure_count = 5
    tft_cb.ato._total_steps = 1000
    tft_cb.ato.checkpoint_times = [10.0] * 2
    tft_cb.ato.step_times = [0.8] * 80
    tft_cb.ato.bind_ckpt_obj(ckpt_cb)

    class DummyException(RuntimeError):
        pass

    with caplog.at_level("WARNING"):
        tft_cb.ato.handle_fault(DummyException("Dummy fault"), 1000)
    clear_env()

    # Test 2: Test fault statistics loading, saving and analysis
    # Set initial data close to threshold
    initial_data = {"failure_count": 10, "total_steps": 1000}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test JSON file in temporary directory
        test_json_path = os.path.join(temp_dir, "ato_info.json")
        with open(test_json_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f)

        # Set environment variable and initialize ATO
        os.environ["MS_ENABLE_TFT"] = f"ATO:1,ATO_THRESHOLD:10,ATO_PATH:{test_json_path}"
        tft_cb = _train_fault_tolerance.TrainFaultTolerance()

        # Verify data loaded correctly
        assert tft_cb.ato._failure_count == 10
        assert tft_cb.ato._total_steps == 1000

        # Set timing data
        tft_cb.ato.checkpoint_times = [10.0] * 2
        tft_cb.ato.step_times = [0.8] * 80
        tft_cb.ato.bind_ckpt_obj(ckpt_cb)

        # Simulate fault to trigger analysis (failure count reaches threshold)
        with patch("mindspore.communication.get_rank", return_value=0):
            tft_cb.ato.handle_fault(DummyException("Dummy fault"), 200)

        # Verify data was accumulated
        assert tft_cb.ato._failure_count == 11  # 10 + 1
        assert tft_cb.ato._total_steps == 1200  # 1000 + 200

        # Verify data was saved to JSON file
        with open(test_json_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["failure_count"] == 11
        assert saved_data["total_steps"] == 1200

    # Verify analysis report was generated
    captured = capsys.readouterr()
    log_content = captured.out + captured.err
    assert "ATO CKPT Recommendations" in log_content
    assert "Failure Rate" in log_content
    assert "Suggested CKPT Interval" in log_content
    assert "Suggested Effective Training Ratio" in log_content

    clear_env()
