# Copyright 2023 Huawei Technologies Co., Ltd
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
Test combination of different configuration manager settings
"""
import pytest

import mindspore._c_dataengine as cde
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# Need to run all these tests in separate processes since tests are modifying config parameters
pytestmark = pytest.mark.forked

ORIGINAL_SEED = 0  # Seed will be set internally in debug mode. Save original seed value to restore.


def setup_function():
    # Save the original seed, since debug mode sets the seed (for deterministic results) if default seed is detected.
    global ORIGINAL_SEED
    ORIGINAL_SEED = ds.config.get_seed()


def teardown_function():
    # Restore the original seed
    ds.config.set_seed(ORIGINAL_SEED)


def create_celeba_pipeline(offload=False, to_pil=False):
    """ Create simple CelebADataset pipeline, suitable for offloading. """
    data1 = ds.CelebADataset("../data/dataset/testCelebAData/")
    data1 = data1.map(operations=[vision.Decode(to_pil=to_pil)], input_columns=["image"])
    data1 = data1.map(operations=[vision.Resize((30, 30))], input_columns=["image"])
    data1 = data1.map(operations=[vision.Rescale(1.0 / 255.0, -1.0),
                                  vision.RandomHorizontalFlip(0.9)],
                      input_columns=["image"], offload=offload)
    data1 = data1.batch(batch_size=2)
    return data1


def iterate_celeba_pipeline(data1):
    """ Iterate over data pipeline created from create_celeba_pipeline. """
    row_count = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        assert item[0].shape == (2, 30, 30, 3)
        assert item[1].shape == (2, 40)
        row_count += 1
    assert row_count == 2


def create_imagefolder_pipeline(data_dir, offload=False, to_pil=False):
    """ Create simple ImageFolderDataset pipeline, suitable for offloading. """
    data1 = ds.ImageFolderDataset(data_dir, num_samples=None, num_parallel_workers=1, shuffle=False)
    data1 = data1.map(operations=[vision.Decode(to_pil=to_pil)], input_columns=["image"])
    data1 = data1.map(operations=[vision.Resize((40, 50))], input_columns=["image"])
    data1 = data1.map(operations=[vision.RandomVerticalFlip(0.8)], input_columns=["image"], offload=offload)
    return data1


def iterate_imagefolder_pipeline(data1):
    """ Iterate over data pipeline created from create_imagefolder_pipeline. """
    row_count = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        row_count += 1
    assert row_count == 6


def run_imagefolder_pyop_pipeline(python_multiprocessing=False, num_epoch=2):
    """ Create and execute ImageFolderDataset pipeline with Python implemented ops. """
    # Define dataset pipeline
    data = ds.ImageFolderDataset("../data/dataset/testImageNetData4/train", decode=True)
    num_repeat = 3
    sample_row = 7
    data = data.repeat(num_repeat)
    # Apply map ops with Python implemented ops
    data = data.map(operations=[(lambda x: x + 1)], input_columns=["image"],
                    python_multiprocessing=python_multiprocessing)
    data = data.map(operations=[vision.RandomErasing(0.6, value='random')], input_columns=["image"],
                    python_multiprocessing=python_multiprocessing)

    # Create and execute iterator
    epoch_count = 0
    sample_count = 0
    iter1 = data.create_dict_iterator(num_epochs=num_epoch)
    for _ in range(num_epoch):
        num_rows = 0
        for row_item in iter1:
            assert len(row_item) == 2
            assert row_item["image"].shape == (384, 682, 3)
            num_rows += 1
        assert num_rows == sample_row * num_repeat
        sample_count += num_rows
        epoch_count += 1
    assert epoch_count == num_epoch
    assert sample_count == num_repeat * num_epoch * sample_row


def test_debug_mode_ignore_config1():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with other configuration API that are ignored
    Expectation: Sanity check of data pipeline is done. Output is equal to the expected output
    """
    # Set specific configurations (to non-default values) which are ignored when debug mode is enabled

    # Reduce memory needed by reducing queue size
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(4)
    # Disable shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)
    # Set number of parallel workers
    num_workers_original = ds.config.get_num_parallel_workers()
    ds.config.set_num_parallel_workers(8)
    # Disable watchdog
    watchdog_original = ds.config.get_enable_watchdog()
    ds.config.set_enable_watchdog(False)
    # Set multiprocessing timeout interval
    mp_timeout_original = ds.config.get_multiprocessing_timeout_interval()
    ds.config.set_multiprocessing_timeout_interval(15)
    # Set monitor interval
    monitor_interval_original = ds.config.get_monitor_sampling_interval()
    ds.config.set_monitor_sampling_interval(10)

    # Enable debug mode after other configurations are set
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)

    # Create and execute data pipeline
    run_imagefolder_pyop_pipeline(python_multiprocessing=False, num_epoch=2)

    # Restore Configuration
    ds.config.set_prefetch_size(prefetch_original)
    ds.config.set_enable_shared_mem(mem_original)
    ds.config.set_num_parallel_workers(num_workers_original)
    ds.config.set_enable_watchdog(watchdog_original)
    ds.config.set_multiprocessing_timeout_interval(mp_timeout_original)
    ds.config.set_monitor_sampling_interval(monitor_interval_original)
    ds.config.set_debug_mode(debug_mode_original)


def test_debug_mode_ignore_config2():
    """
    Feature: Debug Mode
    Description: Test Debug Mode enabled with other configuration API that are ignored
    Expectation: Sanity check of data pipeline is done. Output is equal to the expected output
    """
    # Set specific configurations (to non-default values) which are ignored when debug mode is enabled

    # Enable debug mode before other configurations are set
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)

    # Enable automatic num_parallel_workers
    auto_num_workers_original = ds.config.get_auto_num_workers()
    ds.config.set_auto_num_workers(True)
    # Enable autotune
    autotune_original = ds.config.get_enable_autotune()
    ds.config.set_enable_autotune(True)
    # Set autotune interval
    autotune_interval_original = ds.config.get_autotune_interval()
    ds.config.set_autotune_interval(5)
    # Set callback timeout
    callback_timeout_original = ds.config.get_callback_timeout()
    ds.config.set_callback_timeout(20)
    # Set fast recovery
    fast_recovery_original = ds.config.get_fast_recovery()
    ds.config.set_fast_recovery(False)

    # Create and execute data pipeline
    run_imagefolder_pyop_pipeline(python_multiprocessing=False, num_epoch=2)

    # Restore Configuration
    ds.config.set_debug_mode(debug_mode_original)
    ds.config.set_auto_num_workers(auto_num_workers_original)
    ds.config.set_enable_autotune(autotune_original)
    ds.config.set_autotune_interval(autotune_interval_original)
    ds.config.set_callback_timeout(callback_timeout_original)
    ds.config.set_fast_recovery(fast_recovery_original)


def test_debug_mode_error_samples_mode():
    """
    Feature: Debug Mode
    Description: Test Debug Mode with Error Samples Mode
    Expectation: Error Samples Mode REPLACE or SKIP are ignored.  Data pipeline runs in debug mode.
    """

    # Enable debug mode before other configurations are set
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)

    error_samples_mode_original = ds.config.get_error_samples_mode()

    # Test 1 - Set Error Samples Mode REPLACE
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.REPLACE)
    # Create and iterate a basic pipeline with error samples
    data1 = create_imagefolder_pipeline("../data/dataset/testImageNetError/Sample1_empty/train")
    with pytest.raises(RuntimeError) as error_info:
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "map operation: [Decode] failed" in str(error_info.value)

    # Test 2 - Set Error Samples Mode SKIP
    ds.config.set_error_samples_mode(ds.config.ErrorSamplesMode.SKIP)
    # Create and iterate a basic pipeline with error samples
    data2 = create_imagefolder_pipeline("../data/dataset/testImageNetError/Sample3_text/train")
    with pytest.raises(RuntimeError) as error_info:
        for _ in data2.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    assert "map operation: [Decode] failed" in str(error_info.value)

    # Restore Configuration
    ds.config.set_debug_mode(debug_mode_original)
    ds.config.set_error_samples_mode(error_samples_mode_original)


def test_debug_mode_auto_offload():
    """
    Feature: Debug Mode
    Description: Test Debug Mode with auto_offload configuration API
    Expectation: auto_offload configuration API is ignored.  Data pipeline runs in debug mode.
    """
    # Test 1A - Create data1 pipeline suitable for offloading
    data1 = create_imagefolder_pipeline("../data/dataset/testImageNetData2/train", offload=False)

    # Enable debug mode
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)

    # Enable auto offloading
    auto_offload_original = ds.config.get_auto_offload()
    ds.config.set_auto_offload(True)

    # Test 1B - Execute data1 pipeline
    iterate_imagefolder_pipeline(data1)

    # Test 2 - Create and execute data2 pipeline suitable for offloading
    data2 = create_celeba_pipeline(offload=False)
    iterate_celeba_pipeline(data2)

    # Restore Configuration
    ds.config.set_debug_mode(debug_mode_original)
    ds.config.set_auto_offload(auto_offload_original)


def test_debug_mode_map_offload():
    """
    Feature: Debug Mode
    Description: Test Debug Mode with Map Op Offload Parameter
    Expectation: Offload parameter in Map op is ignored.  Data pipeline runs in debug mode.
    """
    # Test 1A - Create data1 pipeline suitable for offloading with Map Op parameter offload=True
    data1 = create_imagefolder_pipeline("../data/dataset/testImageNetData2/train", offload=True)

    # Enable debug mode
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)

    # Test 1B - Execute data1 pipeline
    iterate_imagefolder_pipeline(data1)

    # Test 2 - Create and execute data2 pipeline suitable for offloading with Map Op parameter offload=True
    data2 = create_celeba_pipeline(offload=True)
    iterate_celeba_pipeline(data2)

    # Restore Configuration
    ds.config.set_debug_mode(debug_mode_original)


def test_debug_mode_profiler():
    """
    Feature: Profiler and Debug Mode
    Description: Test Debug Mode enabled before Profiler
    Expectation: Profiler not enabled.  Data pipeline runs in debug mode.
    """
    # Enable debug mode before profiler is started
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)
    md_profiler = cde.GlobalContext.profiling_manager()
    md_profiler.init()
    md_profiler.start()

    # Create and iterate a basic pipeline
    data1 = create_celeba_pipeline()
    iterate_celeba_pipeline(data1)

    # Restore Configuration
    md_profiler.stop()
    ds.config.set_debug_mode(debug_mode_original)


def test_profiler_debug_mode():
    """
    Feature: Debug Mode and Profiler
    Description: Test Debug Mode enabled after Profiler
    Expectation: Profiler forcibly reset.  Data pipeline runs in debug mode.
    """
    # Enable debug mode after profiler is started
    md_profiler = cde.GlobalContext.profiling_manager()
    md_profiler.init()
    md_profiler.start()
    debug_mode_original = ds.config.get_debug_mode()
    ds.config.set_debug_mode(True)

    # Create and iterate a basic pipeline
    data1 = create_celeba_pipeline()
    iterate_celeba_pipeline(data1)

    # Restore Configuration
    ds.config.set_debug_mode(debug_mode_original)
    md_profiler.stop()


if __name__ == '__main__':
    test_debug_mode_ignore_config1()
    test_debug_mode_ignore_config2()
    test_debug_mode_error_samples_mode()
    test_debug_mode_auto_offload()
    test_debug_mode_map_offload()
    test_debug_mode_profiler()
    test_profiler_debug_mode()
