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
Test OBSMindDataset operations
"""
import os
import sys
import types

import pytest

from mindspore.dataset.engine import datasets_standard_format
from mindspore.dataset.engine.datasets_standard_format import OBSMindDataset
from mindspore.dataset.engine.obs.config_loader import config
from mindspore.dataset.engine.obs import obs_mindrecord_dataset
from mindspore import log as logger

DATA_DIR = ["s3://dataset/imagenet0", "s3://dataset/imagenet1"]


class DummyObsClient:  # pylint: disable=too-few-public-methods
    """Minimal ObsClient stub for util module unit tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


def import_obs_util(monkeypatch):
    """Import obs util with a stub obs SDK for unit tests."""
    fake_obs_module = types.SimpleNamespace(ObsClient=DummyObsClient)
    monkeypatch.setitem(sys.modules, "obs", fake_obs_module)
    monkeypatch.setitem(config.config, "AK", "test-ak")
    monkeypatch.setitem(config.config, "SK", "test-sk")
    monkeypatch.setitem(config.config, "SERVER", "https://test-server")
    sys.modules.pop("mindspore.dataset.engine.obs.util", None)
    from mindspore.dataset.engine.obs import util as obs_util
    return obs_util


def test_obs_mindrecord_exception():
    """
    Feature: Test OBSMindDataset.
    Description: Invalid input.
    Expectation: Raise exception.
    """

    logger.info("Test error cases for MnistDataset")
    error_msg_0 = "Argument dataset_files"
    with pytest.raises(TypeError, match=error_msg_0):
        OBSMindDataset("err_dataset", "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_0_1 = "Item of dataset files"
    with pytest.raises(TypeError, match=error_msg_0_1):
        OBSMindDataset([1, 2], "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_1 = "Argument server"
    with pytest.raises(TypeError, match=error_msg_1):
        OBSMindDataset(DATA_DIR, 12, "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_1_1 = "server should"
    with pytest.raises(ValueError, match=error_msg_1_1):
        OBSMindDataset(DATA_DIR, "ftp://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir")

    error_msg_2 = "Argument ak"
    with pytest.raises(TypeError, match=error_msg_2):
        OBSMindDataset(DATA_DIR, "https://dummy_site", 12, "dummy_sk", "s3://dummy_sync_dir")

    error_msg_3 = "Argument sk"
    with pytest.raises(TypeError, match=error_msg_3):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak", 12, "s3://dummy_sync_dir")

    error_msg_4 = "Argument sync_obs_path"
    with pytest.raises(TypeError, match=error_msg_4):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak", "dummy_sk", 12)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", num_shards=4, shard_id=4)
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", num_shards=2, shard_id=4)

    error_msg_7 = "Argument shard_equal_rows"
    with pytest.raises(TypeError, match=error_msg_7):
        OBSMindDataset(DATA_DIR, "https://dummy_site", "dummy_ak",
                       "dummy_sk", "s3://dummy_sync_dir", shard_equal_rows=1)


def test_obs_mindrecord_uses_instance_config_without_mutating_global_config(monkeypatch):
    """
    Feature: OBSMindDataset
    Description: Each dataset instance should carry its own OBS config instead of mutating module-global config
    Expectation: Global config remains unchanged and per-instance config is passed to backend loader
    """

    class DummyMindRecordFromOBS:
        instances = []

        def __init__(self, dataset_files, columns_list, shuffle, num_shards, shard_id,
                     shard_equal_rows, obs_config):
            self.dataset_files = dataset_files
            self.columns_list = columns_list
            self.shuffle = shuffle
            self.num_shards = num_shards
            self.shard_id = shard_id
            self.shard_equal_rows = shard_equal_rows
            self.obs_config = obs_config
            self.local_path = obs_config.DATASET_LOCAL_PATH
            DummyMindRecordFromOBS.instances.append(self)

        def get_col_names(self):
            return ["image"]

        def __iter__(self):
            return iter(())

    def fake_generator_init(self, *args, **kwargs):
        self.source = kwargs.get("source")
        self.column_names = kwargs.get("column_names")

    original_config = dict(config.config)

    monkeypatch.setattr(datasets_standard_format, "MindRecordFromOBS", DummyMindRecordFromOBS)
    monkeypatch.setattr(datasets_standard_format.GeneratorDataset, "__init__", fake_generator_init)

    OBSMindDataset(DATA_DIR, "https://site-one", "ak-one", "sk-one", "s3://sync-one")
    OBSMindDataset(DATA_DIR, "https://site-two", "ak-two", "sk-two", "s3://sync-two")

    assert len(DummyMindRecordFromOBS.instances) == 2
    assert DummyMindRecordFromOBS.instances[0].obs_config is not None
    assert DummyMindRecordFromOBS.instances[1].obs_config is not None
    assert DummyMindRecordFromOBS.instances[0].obs_config.AK == "ak-one"
    assert DummyMindRecordFromOBS.instances[1].obs_config.AK == "ak-two"
    assert DummyMindRecordFromOBS.instances[0].obs_config.SK == "sk-one"
    assert DummyMindRecordFromOBS.instances[1].obs_config.SK == "sk-two"
    assert DummyMindRecordFromOBS.instances[0].obs_config.SERVER == "https://site-one"
    assert DummyMindRecordFromOBS.instances[1].obs_config.SERVER == "https://site-two"
    assert DummyMindRecordFromOBS.instances[0].obs_config.SYNC_OBS_PATH == "s3://sync-one"
    assert DummyMindRecordFromOBS.instances[1].obs_config.SYNC_OBS_PATH == "s3://sync-two"
    assert config.config == original_config


def test_minddataset_normalizes_windows_list_paths(monkeypatch):
    """
    Feature: MindDataset
    Description: Normalize Windows-style path separators for list inputs
    Expectation: Returned paths use forward slashes
    """
    dataset = datasets_standard_format.MindDataset.__new__(datasets_standard_format.MindDataset)
    monkeypatch.setattr(datasets_standard_format.platform, "system", lambda: "Windows")

    # pylint: disable=protected-access
    normalized_files = dataset._normalize_dataset_files(
        [r"C:\tmp\data.mindrecord", r"D:\meta\file.mindrecord"])

    assert normalized_files == ["C:/tmp/data.mindrecord", "D:/meta/file.mindrecord"]


def test_make_dataset_tuple_uses_read_only_sqlite(monkeypatch, tmp_path):
    """
    Feature: OBS util
    Description: Read metadata size with a read-only sqlite connection
    Expectation: sqlite URI uses mode=ro and the connection context is closed
    """
    obs_util = import_obs_util(monkeypatch)
    captured = {}

    class DummyConnection:
        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, traceback):
            captured["exited"] = True
            return False

        def execute(self, query):
            captured["query"] = query
            return [(7,)]

    def fake_connect(database_uri, **kwargs):
        captured["uri"] = database_uri
        captured["uri_flag"] = kwargs.get("uri")
        return DummyConnection()

    monkeypatch.setattr(obs_util.sqlite3, "connect", fake_connect)

    dataset_size, file_sizes = obs_util.make_dataset_tuple(["s3://bucket/data.mindrecord"], str(tmp_path))

    assert dataset_size == 7
    assert file_sizes == [("s3://bucket/data.mindrecord", 7)]
    assert captured["uri"] == "file:{}?mode=ro".format(os.path.join(str(tmp_path), "data.mindrecord.db"))
    assert captured["uri_flag"] is True
    assert captured["query"] == "SELECT COUNT(*) FROM INDEXES"
    assert captured["entered"] is True
    assert captured["exited"] is True


def test_exclusive_lock_uses_hashed_lock_file_path(monkeypatch, tmp_path):
    """
    Feature: OBS util
    Description: Derive deterministic lock file names without exposing the raw lock key
    Expectation: Lock file path lives under the dedicated lock directory and does not contain the raw name
    """
    obs_util = import_obs_util(monkeypatch)
    monkeypatch.setattr(obs_util, "LOCK_FILE_DIR", str(tmp_path / "locks"))

    # pylint: disable=protected-access
    lock_path = obs_util._get_lock_file_path("raw-lock-name")
    same_lock_path = obs_util._get_lock_file_path("raw-lock-name")
    other_lock_path = obs_util._get_lock_file_path("another-lock-name")

    assert os.path.dirname(lock_path) == str(tmp_path / "locks")
    assert lock_path == same_lock_path
    assert lock_path != other_lock_path
    assert "raw-lock-name" not in os.path.basename(lock_path)
    assert os.path.isdir(obs_util.LOCK_FILE_DIR)


def test_remove_unused_dataset_handles_raced_entries(monkeypatch):
    """
    Feature: OBSMindDataset cleanup
    Description: Ignore raced-away files and directory entries during cleanup
    Expectation: Cleanup completes without raising when unlink sees missing files or directories
    """
    local_path = "/tmp/local_dataset"
    sync_dir = "/cache/sync_data/1"
    local_path_fd = 123
    unlinked = []

    def fake_exists(path):
        return path == sync_dir

    def fake_listdir(path):
        if path == sync_dir:
            return ["ready_1", "ready_2", "ready_3", "ready_4", "ready_5", "ready_6", "ready_7"]
        if path == local_path:
            return ["keep.db", "gone.mindrecord", "subdir"]
        return []

    def fake_unlink(path, dir_fd=None):
        unlinked.append((path, dir_fd))
        if path == "gone.mindrecord":
            raise FileNotFoundError()
        if path == "subdir":
            raise IsADirectoryError()

    monkeypatch.setattr(obs_mindrecord_dataset.os, "path", obs_mindrecord_dataset.os.path)
    monkeypatch.setattr(obs_mindrecord_dataset.os.path, "exists", fake_exists)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "listdir", fake_listdir)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "open", lambda path, flags: local_path_fd)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "close", lambda fd: None)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "unlink", fake_unlink)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "remove", lambda path: None)

    # pylint: disable=protected-access
    obs_mindrecord_dataset._remove_unused_dataset(
        local_path, num_shards=16, shard_id=0, epoch_num=1)

    assert ("gone.mindrecord", local_path_fd) in unlinked
    assert ("subdir", local_path_fd) in unlinked


if __name__ == '__main__':
    test_obs_mindrecord_exception()
