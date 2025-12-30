"""
Configuration file for pytest fixtures used in test_internal_ops.
"""
import pytest
import os


@pytest.fixture(scope="function", autouse=False)
def enable_asd_env_fixture():
    old = os.environ.get("MS_ENABLE_INTERNAL_BOOST")
    old_matmul = os.environ.get("MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST")
    os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"
    os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "MatMulElemwise"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MS_ENABLE_INTERNAL_BOOST", None)
        else:
            os.environ["MS_ENABLE_INTERNAL_BOOST"] = old
        if old_matmul is None:
            os.environ.pop("MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST", None)
        else:
            os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = old_matmul


@pytest.fixture(scope="function", autouse=False)
def enable_internal_env_fixture():
    ms_sync_on = os.environ.get("MS_SYNC_RUN")
    os.environ["MS_SYNC_RUN"] = "on"
    old = os.environ.get("MS_ENABLE_INTERNAL_BOOST")
    if old is None:
        os.environ.pop("MS_ENABLE_INTERNAL_BOOST", None)
    else:
        os.environ.pop("MS_ENABLE_INTERNAL_BOOST", old)
    if ms_sync_on is None:
        os.environ.pop("MS_SYNC_RUN", None)
    else:
        os.environ.pop("MS_SYNC_RUN", ms_sync_on)
