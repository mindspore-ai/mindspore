# Copyright 2022-2026 Huawei Technologies Co., Ltd
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

import importlib
import os
import sqlite3
import sys
import threading
import time as pytime
import types
import uuid

import pytest

from mindspore import log as logger
from mindspore.dataset.engine import datasets_standard_format
from mindspore.dataset.engine.datasets_standard_format import OBSMindDataset
from mindspore.dataset.engine.obs import obs_mindrecord_dataset
from mindspore.dataset.engine.obs.config_loader import create_obs_config

DATA_DIR = ["s3://dataset/imagenet0", "s3://dataset/imagenet1"]
_REAL_SLEEP = pytime.sleep


class DummyObsClient:  # pylint: disable=too-few-public-methods
    """Minimal ObsClient stub for util module unit tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeObsService:
    """In-memory fake OBS backend for integration-like unit tests."""

    def __init__(self):
        self._objects = {}
        self._bucket_auth = {}
        self.client_inits = []
        self.download_records = []
        self._lock = threading.Lock()

    def set_bucket_auth(self, bucket, credentials=None):
        """Set bucket credential policy. None means no auth check."""
        self._bucket_auth[bucket] = credentials

    def register_client(self, access_key_id, secret_access_key, server):
        """Record initialized fake OBS clients for assertions."""
        with self._lock:
            self.client_inits.append((access_key_id, secret_access_key, server))

    def is_authorized(self, bucket, access_key_id, secret_access_key):
        """Check client credentials for a bucket."""
        expected = self._bucket_auth.get(bucket)
        if expected is None:
            return True
        return expected == (access_key_id, secret_access_key)

    def has_bucket(self, bucket):
        """Return whether bucket has at least one object."""
        with self._lock:
            return any(stored_bucket == bucket for stored_bucket, _ in self._objects)

    def has_object_metadata(self, bucket, key):
        """Return whether object exists."""
        with self._lock:
            return (bucket, key) in self._objects

    def fetch_object(self, bucket, key):
        """Get object content by key."""
        with self._lock:
            return self._objects.get((bucket, key))

    def append_download_record(self, bucket, key, access_key_id, secret_access_key):
        """Record one download request."""
        with self._lock:
            self.download_records.append((bucket, key, access_key_id, secret_access_key))

    def put_object(self, bucket, key, content=b""):
        with self._lock:
            self._objects[(bucket, key)] = content

    def list_object_keys(self, bucket, prefix):
        """List object keys with prefix."""
        with self._lock:
            return [key for stored_bucket, key in self._objects if stored_bucket == bucket and key.startswith(prefix)]

    def has_object(self, bucket, key):
        with self._lock:
            return (bucket, key) in self._objects

    def build_client_class(self):
        """Build a Fake ObsClient class bound to current service."""
        service = self

        class _FakeObsClient:  # pylint: disable=too-few-public-methods
            def __init__(self, access_key_id, secret_access_key, server):
                self.access_key_id = access_key_id
                self.secret_access_key = secret_access_key
                self.server = server
                self.port = 443
                service.register_client(access_key_id, secret_access_key, server)

            def _authorized(self, bucket):
                return service.is_authorized(bucket, self.access_key_id, self.secret_access_key)

            @staticmethod
            def _resp(status, body=None, error_code="", error_message=""):
                return types.SimpleNamespace(
                    status=status,
                    requestId="fake-request-id",
                    body=body,
                    errorCode=error_code,
                    errorMessage=error_message,
                )

            def headBucket(self, bucket_name):
                has_bucket = service.has_bucket(bucket_name)
                if not self._authorized(bucket_name):
                    return self._resp(403, error_code="AccessDenied", error_message="Forbidden")
                if has_bucket:
                    return self._resp(200)
                return self._resp(404, error_code="NoSuchBucket", error_message="Bucket not found")

            def getObjectMetadata(self, bucket_name, object_key):
                if not self._authorized(bucket_name):
                    return self._resp(403, error_code="AccessDenied", error_message="Forbidden")
                exists = service.has_object_metadata(bucket_name, object_key)
                if exists:
                    return self._resp(200)
                return self._resp(404, error_code="NoSuchKey", error_message="Object not found")

            def downloadFile(self, bucket_name, object_key, local_path, _part_size, _task_num, _enable_checkpoint):
                if not self._authorized(bucket_name):
                    return self._resp(403, error_code="AccessDenied", error_message="Forbidden")
                content = service.fetch_object(bucket_name, object_key)
                service.append_download_record(bucket_name, object_key, self.access_key_id, self.secret_access_key)
                if content is None:
                    return self._resp(404, error_code="NoSuchKey", error_message="Object not found")
                parent = os.path.dirname(local_path)
                if parent and not os.path.exists(parent):
                    os.makedirs(parent, exist_ok=True)
                if isinstance(content, str):
                    content = content.encode("utf-8")
                with open(local_path, "wb") as f:
                    f.write(content)
                return self._resp(200)

            def putContent(self, bucket_name, object_key, content=None):
                if not self._authorized(bucket_name):
                    return self._resp(403, error_code="AccessDenied", error_message="Forbidden")
                data = b"" if content is None else content
                if isinstance(data, str):
                    data = data.encode("utf-8")
                service.put_object(bucket_name, object_key, data)
                return self._resp(200)

            def listObjects(self, bucket_name, prefix):
                if not self._authorized(bucket_name):
                    return self._resp(403, error_code="AccessDenied", error_message="Forbidden")
                keys = service.list_object_keys(bucket_name, prefix)
                contents = [types.SimpleNamespace(key=key) for key in keys]
                body = types.SimpleNamespace(contents=contents)
                return self._resp(200, body=body)

            def close(self):
                pass

        return _FakeObsClient


def _create_meta_db_bytes(tmp_path, row_count):
    """Create sqlite metadata bytes with INDEXES table."""
    db_path = tmp_path / f"fake_meta_{row_count}_{uuid.uuid4().hex}.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE INDEXES(id INTEGER)")
        conn.executemany("INSERT INTO INDEXES(id) VALUES(?)", [(i,) for i in range(row_count)])
        conn.commit()
    finally:
        conn.close()
    with open(db_path, "rb") as f:
        return f.read()


def _setup_fake_obs_e2e_env(monkeypatch, tmp_path, service):
    """Patch runtime environment for fake OBS end-to-end tests."""
    from mindspore.dataset.engine.obs import util as obs_util

    fake_client_cls = service.build_client_class()
    monkeypatch.setattr(obs_util, "ObsClient", fake_client_cls)
    monkeypatch.setitem(obs_mindrecord_dataset.OBSRuntime.get_obs_client.__globals__, "ObsClient", fake_client_cls)
    monkeypatch.setenv("DATASET_LOCAL_PATH", str(tmp_path / "dataset_local"))
    monkeypatch.setenv("WORKING_PATH", str(tmp_path / "work"))
    monkeypatch.setenv("WAIT_STEP_TIME", "0")
    monkeypatch.setenv("WAIT_META_TIME", "0")
    monkeypatch.setenv("WARMINGUP_TIME", "0")
    monkeypatch.setenv("RETRY_DELTA_TIME", "0")
    monkeypatch.setenv("MAX_RETRY", "3")
    monkeypatch.setenv("BATCH_JOB_ID", "job-ut")
    monkeypatch.setattr(obs_mindrecord_dataset.time, "sleep", lambda *_args, **_kwargs: _REAL_SLEEP(0.001))

    class FakeMindDataset:
        def __init__(self, dataset_files, columns_list=None, sampler=None, shuffle=None):
            self.dataset_files = dataset_files
            self.columns_list = columns_list
            self.sampler = sampler
            self.shuffle = shuffle

        def create_tuple_iterator(self, num_epochs=1, output_numpy=True):  # pylint: disable=unused-argument
            name = os.path.basename(self.dataset_files[0])
            return iter([(name,)])

        def get_col_names(self):
            return ["data"]

    def fake_generator_init(self, *args, **kwargs):
        self.source = kwargs.get("source")
        self.column_names = kwargs.get("column_names")

    monkeypatch.setattr(datasets_standard_format, "MindDataset", FakeMindDataset)
    monkeypatch.setattr(datasets_standard_format.GeneratorDataset, "__init__", fake_generator_init)


def import_obs_util(monkeypatch):
    """Import obs util with a stub obs SDK for unit tests."""
    fake_obs_module = types.SimpleNamespace(ObsClient=DummyObsClient)
    monkeypatch.setitem(sys.modules, "obs", fake_obs_module)
    module_name = "mindspore.dataset.engine.obs.util"
    sys.modules.pop(module_name, None)
    obs_pkg = importlib.import_module("mindspore.dataset.engine.obs")
    if hasattr(obs_pkg, "util"):
        delattr(obs_pkg, "util")
    return importlib.import_module(module_name)


def import_obs_util_without_global_credentials(monkeypatch):
    """Import obs util without providing runtime OBS credentials."""
    fake_obs_module = types.SimpleNamespace(ObsClient=DummyObsClient)
    monkeypatch.setitem(sys.modules, "obs", fake_obs_module)
    module_name = "mindspore.dataset.engine.obs.util"
    sys.modules.pop(module_name, None)
    obs_pkg = importlib.import_module("mindspore.dataset.engine.obs")
    if hasattr(obs_pkg, "util"):
        delattr(obs_pkg, "util")
    return importlib.import_module(module_name)


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
        OBSMindDataset(
            DATA_DIR, "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir", num_shards=2, shard_id=-1
        )
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(
            DATA_DIR, "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir", num_shards=4, shard_id=4
        )
    with pytest.raises(ValueError, match=error_msg_5):
        OBSMindDataset(
            DATA_DIR, "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir", num_shards=2, shard_id=4
        )

    error_msg_7 = "Argument shard_equal_rows"
    with pytest.raises(TypeError, match=error_msg_7):
        OBSMindDataset(
            DATA_DIR, "https://dummy_site", "dummy_ak", "dummy_sk", "s3://dummy_sync_dir", shard_equal_rows=1
        )


def test_obs_mindrecord_uses_instance_config_without_mutating_global_config(monkeypatch):
    """
    Feature: OBSMindDataset
    Description: Each dataset instance should carry an isolated OBS config object
    Expectation: Different instances hold different config objects and expected credentials
    """

    class DummyMindRecordFromOBS:
        instances = []

        def __init__(self, dataset_files, columns_list, shuffle, num_shards, shard_id, shard_equal_rows, obs_config):
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
    assert DummyMindRecordFromOBS.instances[0].obs_config is not DummyMindRecordFromOBS.instances[1].obs_config


def test_obs_util_import_does_not_require_global_obs_credentials(monkeypatch):
    """
    Feature: OBS util
    Description: Importing the util module should not eagerly require global AK/SK/SERVER values
    Expectation: Module import succeeds and OBS client creation is deferred until runtime
    """
    obs_util = import_obs_util_without_global_credentials(monkeypatch)

    assert hasattr(obs_util, "OBSRuntime")
    runtime = obs_util.OBSRuntime(create_obs_config())
    with pytest.raises(RuntimeError, match="Variable AK can not be found"):
        runtime.get_obs_client()


def test_get_obs_client_uses_explicit_obs_config(monkeypatch):
    """
    Feature: OBS util
    Description: Build the OBS client from an explicit config object instead of module-global secrets
    Expectation: ObsClient receives the explicit AK/SK/SERVER values
    """
    obs_util = import_obs_util_without_global_credentials(monkeypatch)
    created_clients = []

    class RecordingObsClient:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            created_clients.append(kwargs)

        def close(self):
            pass

    explicit_config = create_obs_config()
    explicit_config.AK = "instance-ak"
    explicit_config.SK = "instance-sk"
    explicit_config.SERVER = "https://instance-server"

    monkeypatch.setattr(obs_util, "ObsClient", RecordingObsClient)

    runtime = obs_util.OBSRuntime(explicit_config)
    runtime.get_obs_client()

    assert created_clients == [
        {"access_key_id": "instance-ak", "secret_access_key": "instance-sk", "server": "https://instance-server"}
    ]


def test_get_obs_client_reuses_runtime_client(monkeypatch):
    """
    Feature: OBS util
    Description: Reuse the OBS client owned by one OBSRuntime instance
    Expectation: Repeated calls return the same client and initialize ObsClient only once
    """
    obs_util = import_obs_util_without_global_credentials(monkeypatch)
    created_clients = []

    class RecordingObsClient:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        def close(self):
            pass

    explicit_config = create_obs_config()
    explicit_config.AK = "instance-ak"
    explicit_config.SK = "instance-sk"
    explicit_config.SERVER = "https://instance-server"

    monkeypatch.setattr(obs_util, "ObsClient", RecordingObsClient)

    runtime = obs_util.OBSRuntime(explicit_config)
    first_client = runtime.get_obs_client()
    second_client = runtime.get_obs_client()

    assert first_client is second_client
    assert created_clients == [first_client]
    assert first_client.kwargs == {
        "access_key_id": "instance-ak",
        "secret_access_key": "instance-sk",
        "server": "https://instance-server",
    }


def test_obs_runtime_close_closes_cached_client(monkeypatch):
    """
    Feature: OBS util
    Description: Close the OBS client cached by one OBSRuntime instance
    Expectation: close() releases the cached client and remains idempotent
    """
    obs_util = import_obs_util_without_global_credentials(monkeypatch)
    close_calls = []

    class RecordingObsClient:  # pylint: disable=too-few-public-methods
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def close(self):
            close_calls.append(self)

    explicit_config = create_obs_config()
    explicit_config.AK = "instance-ak"
    explicit_config.SK = "instance-sk"
    explicit_config.SERVER = "https://instance-server"

    monkeypatch.setattr(obs_util, "ObsClient", RecordingObsClient)

    runtime = obs_util.OBSRuntime(explicit_config)
    obs_client = runtime.get_obs_client()
    runtime.close()
    runtime.close()

    assert close_calls == [obs_client]
    assert runtime.obs_client is None


def test_minddataset_normalizes_windows_list_paths(monkeypatch):
    """
    Feature: MindDataset
    Description: Normalize Windows-style path separators for list inputs
    Expectation: Returned paths use forward slashes
    """
    dataset = datasets_standard_format.MindDataset.__new__(datasets_standard_format.MindDataset)
    monkeypatch.setattr(datasets_standard_format.platform, "system", lambda: "Windows")

    # pylint: disable=protected-access
    normalized_files = dataset._normalize_dataset_files([r"C:\tmp\data.mindrecord", r"D:\meta\file.mindrecord"])

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


def test_init_shard_files_uses_instance_config_for_meta_fetch(monkeypatch):
    """
    Feature: OBSMindDataset
    Description: Metadata prefetch should use the per-instance OBS runtime
    Expectation: fetch_meta_files is called on the runtime object passed to shard init
    """
    captured = {}
    obs_config = create_obs_config()
    obs_config.AK = "instance-ak"
    obs_config.SK = "instance-sk"
    obs_config.SERVER = "https://instance-server"
    obs_config.SYNC_OBS_PATH = "s3://bucket/sync"
    import_obs_util(monkeypatch)

    class DummyRuntime:
        def __init__(self, runtime_config):
            self.config = runtime_config

        def fetch_meta_files(self, dataset_files, local_path):
            captured["dataset_files"] = list(dataset_files)
            captured["local_path"] = local_path
            captured["runtime"] = self

    runtime = DummyRuntime(obs_config)
    monkeypatch.setattr(
        obs_mindrecord_dataset, "make_dataset_tuple", lambda dataset_files, local_path: (6, [(dataset_files[0], 6)])
    )
    monkeypatch.setattr(obs_mindrecord_dataset.random, "shuffle", lambda dataset_files: None)

    # pylint: disable=protected-access
    shard_files, size_per_shard = obs_mindrecord_dataset._init_shard_files(
        DATA_DIR[:1],
        shuffle=False,
        seed=1,
        num_shards=1,
        shard_id=0,
        shard_equal_rows=True,
        size_per_shard=None,
        local_path="/tmp/local_dataset",
        current_epoch=1,
        obs_runtime=runtime,
    )

    assert shard_files == [(DATA_DIR[0], 0, 6, True)]
    assert size_per_shard == 6
    assert captured["runtime"] is runtime
    assert captured["dataset_files"] == DATA_DIR[:1]
    assert captured["local_path"] == "/tmp/local_dataset"


def test_download_work_uses_instance_config():
    """
    Feature: OBSMindDataset
    Description: Background downloads should use the per-instance OBS runtime
    Expectation: try_load_from_obs is called on the runtime object passed to downloader
    """
    captured = {}
    obs_config = create_obs_config()
    obs_config.AK = "instance-ak"
    obs_config.SK = "instance-sk"
    obs_config.SERVER = "https://instance-server"
    obs_config.WORKING_PATH = "/tmp/obs_working_path"

    class OneShotQueue:
        def get(self):
            return (0, "s3://bucket/data.mindrecord")

    class DummyRuntime:
        def __init__(self, runtime_config):
            self.config = runtime_config

        def get_used_disk_per(self):
            return 0.0

        def try_load_from_obs(self, remote_path, dataset_file, local_path):
            captured["remote_path"] = remote_path
            captured["dataset_file"] = dataset_file
            captured["local_path"] = local_path
            captured["runtime"] = self
            raise RuntimeError("stop-download-loop")

    runtime = DummyRuntime(obs_config)

    with pytest.raises(RuntimeError, match="stop-download-loop"):
        # pylint: disable=protected-access
        obs_mindrecord_dataset._download_work(
            shard_id=0,
            current_idx=types.SimpleNamespace(value=0),
            local_path="/tmp/local_dataset",
            cache={"data.mindrecord": (0, False)},
            q=OneShotQueue(),
            obs_runtime=runtime,
        )

    assert captured["remote_path"] == "s3://bucket"
    assert captured["dataset_file"] == "data.mindrecord"
    assert captured["local_path"] == "/tmp/local_dataset"
    assert captured["runtime"] is runtime


def test_mindrecord_from_obs_len_returns_cached_size(monkeypatch):
    """
    Feature: MindRecordFromOBS
    Description: __len__ should return the computed shard size directly
    Expectation: No recursive self-call and size is cached on the instance
    """
    obs_mindrecord_dataset_module = obs_mindrecord_dataset
    captured = {}

    class DummyRuntime:
        def __init__(self):
            self.config = types.SimpleNamespace()

        def fetch_meta_files(self, dataset_files, local_path):
            captured["dataset_files"] = list(dataset_files)
            captured["local_path"] = local_path

        def close(self):
            pass

    monkeypatch.setattr(
        obs_mindrecord_dataset_module,
        "make_dataset_tuple",
        lambda dataset_files, local_path: (6, [(dataset_files[0], 6)]),
    )

    # pylint: disable=protected-access
    dataset = obs_mindrecord_dataset_module.MindRecordFromOBS.__new__(obs_mindrecord_dataset_module.MindRecordFromOBS)
    dataset._size_per_shard = None
    dataset._shard_files = [("s3://bucket/data.mindrecord", 0, 6, True)]
    dataset._local_path = "/tmp/local_dataset"
    dataset._obs_runtime = DummyRuntime()

    assert len(dataset) == 6
    assert dataset._size_per_shard == 6
    assert captured["dataset_files"] == ["s3://bucket/data.mindrecord"]
    assert captured["local_path"] == "/tmp/local_dataset"


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


def test_sync_wait_for_dataset_uses_instance_config(monkeypatch):
    """
    Feature: OBSMindDataset
    Description: Sync polling should use the per-instance OBS runtime
    Expectation: The OBS client from runtime polls the configured sync path
    """
    captured = {}
    obs_config = create_obs_config()
    obs_config.AK = "instance-ak"
    obs_config.SK = "instance-sk"
    obs_config.SERVER = "https://instance-server"
    obs_config.SYNC_OBS_PATH = "s3://bucket/sync_dir"

    class RecordingClient:  # pylint: disable=too-few-public-methods
        def listObjects(self, bucket_name, prefix):
            captured["bucket_name"] = bucket_name
            captured["prefix"] = prefix
            return types.SimpleNamespace(
                status=200, body=types.SimpleNamespace(contents=[types.SimpleNamespace(key=prefix + "ready_0.txt")])
            )

    class DummyRuntime:
        def __init__(self, runtime_config):
            self.config = runtime_config

        def get_obs_client(self):
            captured["runtime"] = self
            return RecordingClient()

    runtime = DummyRuntime(obs_config)
    monkeypatch.setattr(obs_mindrecord_dataset.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("BATCH_JOB_ID", "job-1")

    obs_mindrecord_dataset.sync_wait_for_dataset(0, 1, 3, obs_runtime=runtime)

    assert captured["runtime"] is runtime
    assert captured["bucket_name"] == "bucket"
    assert captured["prefix"] == "sync_dir/job-1/3/"


def test_sync_wait_for_dataset_requires_initialized_runtime():
    """
    Feature: OBSMindDataset
    Description: Sync polling should require explicit runtime
    Expectation: Raise TypeError when runtime is not passed
    """
    # pylint: disable=no-value-for-parameter
    with pytest.raises(TypeError, match="obs_runtime"):
        obs_mindrecord_dataset.sync_wait_for_dataset(0, 1, 1)


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
            raise FileNotFoundError
        if path == "subdir":
            raise IsADirectoryError

    monkeypatch.setattr(obs_mindrecord_dataset.os, "path", obs_mindrecord_dataset.os.path)
    monkeypatch.setattr(obs_mindrecord_dataset.os.path, "exists", fake_exists)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "listdir", fake_listdir)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "open", lambda path, flags: local_path_fd)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "close", lambda fd: None)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "unlink", fake_unlink)
    monkeypatch.setattr(obs_mindrecord_dataset.os, "remove", lambda path: None)
    runtime = types.SimpleNamespace(config=types.SimpleNamespace(WARMINGUP_TIME=0))

    # pylint: disable=protected-access
    obs_mindrecord_dataset._remove_unused_dataset(
        local_path, num_shards=16, shard_id=0, epoch_num=1, obs_runtime=runtime
    )

    assert ("gone.mindrecord", local_path_fd) in unlinked
    assert ("subdir", local_path_fd) in unlinked


def test_wait_remove_dataset_uses_runtime_config(monkeypatch):
    """
    Feature: OBSMindDataset cleanup
    Description: Wait for dataset removal with the runtime config object
    Expectation: The wait path reads WARMINGUP_TIME directly from obs_runtime.config
    """
    sync_dir = "/cache/sync_data/2"
    sync_file = os.path.join(sync_dir, "ready_1")
    sleep_calls = []
    listdir_results = iter([["ready_0"], []])

    class FakeFile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, content):
            assert content == "ok"

    def fake_sleep(value):
        sleep_calls.append(value)

    monkeypatch.setattr(obs_mindrecord_dataset.os.path, "exists", lambda path: path in (sync_dir, sync_file))
    monkeypatch.setattr(obs_mindrecord_dataset.os, "listdir", lambda _path: next(listdir_results))
    monkeypatch.setattr(obs_mindrecord_dataset.os, "chmod", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(obs_mindrecord_dataset.time, "sleep", fake_sleep)
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: FakeFile())
    runtime = types.SimpleNamespace(config=types.SimpleNamespace(WARMINGUP_TIME=0.25))

    # pylint: disable=protected-access
    obs_mindrecord_dataset._wait_remove_dataset(num_shards=16, shard_id=1, epoch_num=2, obs_runtime=runtime)

    assert sleep_calls == [0.25]


def test_mindrecord_from_obs_close_releases_owned_resources_in_order():
    """
    Feature: OBSMindDataset cleanup
    Description: Close dataset-owned resources after stopping the download pool
    Expectation: close() terminates the pool, shuts down the manager, then closes the runtime once
    """
    close_order = []

    class FakePool:
        def terminate(self):
            close_order.append("pool")

    class FakeManager:
        def shutdown(self):
            close_order.append("manager")

    class FakeRuntime:
        def close(self):
            close_order.append("runtime")

    source = obs_mindrecord_dataset.MindRecordFromOBS.__new__(obs_mindrecord_dataset.MindRecordFromOBS)
    source._pool = FakePool()  # pylint: disable=protected-access
    source._manager = FakeManager()  # pylint: disable=protected-access
    source._queue = object()  # pylint: disable=protected-access
    source._cache = object()  # pylint: disable=protected-access
    source._current_idx = object()  # pylint: disable=protected-access
    source._obs_runtime = FakeRuntime()  # pylint: disable=protected-access

    source.close()
    source.close()

    assert close_order == ["pool", "manager", "runtime"]
    assert source._pool is None  # pylint: disable=protected-access
    assert source._manager is None  # pylint: disable=protected-access
    assert source._queue is None  # pylint: disable=protected-access
    assert source._cache is None  # pylint: disable=protected-access
    assert source._current_idx is None  # pylint: disable=protected-access
    assert source._obs_runtime is None  # pylint: disable=protected-access


def test_obsminddataset_e2e_with_fake_obs_service(monkeypatch, tmp_path):
    """
    Feature: OBSMindDataset e2e with fake OBS
    Description: Run OBSMindDataset main flow against an in-memory OBS backend
    Expectation: Metadata/data are downloaded, synchronization marker is uploaded, and data can be iterated
    """
    service = FakeObsService()
    _setup_fake_obs_e2e_env(monkeypatch, tmp_path, service)

    data_bucket = "data-bucket"
    sync_bucket = "sync-bucket"
    dataset_key = "records/train.mindrecord"
    dataset_uri = f"s3://{data_bucket}/{dataset_key}"
    sync_uri = f"s3://{sync_bucket}/sync-root"
    service.set_bucket_auth(data_bucket, ("ak-single", "sk-single"))
    service.set_bucket_auth(sync_bucket, None)
    service.put_object(data_bucket, dataset_key, b"mindrecord-data")
    service.put_object(data_bucket, dataset_key + ".db", _create_meta_db_bytes(tmp_path, 3))
    service.put_object(sync_bucket, "sync-root/", b"")

    dataset = OBSMindDataset(
        [dataset_uri],
        "https://fake-server",
        "ak-single",
        "sk-single",
        sync_uri,
        shuffle=False,
        num_shards=1,
        shard_id=0,
    )
    source = dataset.source
    try:
        sample = next(iter(source))
        assert sample == ("train.mindrecord",)
    finally:
        source.close()

    local_root = os.path.join(str(tmp_path), "dataset_local")
    assert os.path.exists(os.path.join(local_root, "train.mindrecord"))
    assert os.path.exists(os.path.join(local_root, "train.mindrecord.db"))
    assert service.has_object(sync_bucket, "sync-root/job-ut/1/download_dataset_ready_0.txt")


def test_obsminddataset_e2e_with_fake_obs_service_isolates_credentials(monkeypatch, tmp_path):
    """
    Feature: OBSMindDataset e2e with fake OBS
    Description: Two OBSMindDataset instances use different credentials against different buckets
    Expectation: Download requests for each bucket always use that bucket's expected AK/SK
    """
    service = FakeObsService()
    _setup_fake_obs_e2e_env(monkeypatch, tmp_path, service)

    service.set_bucket_auth("bucket-a", ("ak-a", "sk-a"))
    service.set_bucket_auth("bucket-b", ("ak-b", "sk-b"))
    service.set_bucket_auth("sync-bucket", None)
    service.put_object("bucket-a", "records/a.mindrecord", b"a-data")
    service.put_object("bucket-a", "records/a.mindrecord.db", _create_meta_db_bytes(tmp_path, 2))
    service.put_object("bucket-b", "records/b.mindrecord", b"b-data")
    service.put_object("bucket-b", "records/b.mindrecord.db", _create_meta_db_bytes(tmp_path, 4))
    service.put_object("sync-bucket", "sync-root/", b"")

    dataset_a = OBSMindDataset(
        ["s3://bucket-a/records/a.mindrecord"],
        "https://fake-server",
        "ak-a",
        "sk-a",
        "s3://sync-bucket/sync-root",
        shuffle=False,
        num_shards=1,
        shard_id=0,
    )
    dataset_b = OBSMindDataset(
        ["s3://bucket-b/records/b.mindrecord"],
        "https://fake-server",
        "ak-b",
        "sk-b",
        "s3://sync-bucket/sync-root",
        shuffle=False,
        num_shards=1,
        shard_id=0,
    )
    source_a = dataset_a.source
    source_b = dataset_b.source
    try:
        assert next(iter(source_a)) == ("a.mindrecord",)
        assert next(iter(source_b)) == ("b.mindrecord",)
    finally:
        source_a.close()
        source_b.close()

    expected = {"bucket-a": ("ak-a", "sk-a"), "bucket-b": ("ak-b", "sk-b")}
    assert service.download_records
    for bucket, access_key, secret_key in ((item[0], item[2], item[3]) for item in service.download_records):
        if bucket in expected:
            assert (access_key, secret_key) == expected[bucket]


if __name__ == "__main__":
    test_obs_mindrecord_exception()
