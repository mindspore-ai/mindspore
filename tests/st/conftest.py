"""
conftest.py for all cases
"""
import os
import pytest
import shutil

if os.path.exists("/usr/local/Ascend/driver") and shutil.which("npu-smi"):
    print("="*50 + "Ascend")
    def byte_to_mb(value: int) -> float:
        """
        Convert bytes to megabytes.

        Args:
            value (int): Value in bytes to convert

        Returns:
            float: Value converted to megabytes
        """
        mem = value / 1024 / 1024
        return round(mem, 2)


    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_teardown(item, nextitem):
        """
        Pytest hook function executed after each test case teardown.
        Handles memory monitoring and NPU resource cleanup.

        Args:
            item: Pytest test item
            nextitem: Next test item to be executed
        """
        marker = item.get_closest_marker("onecard")
        if marker:
            try:
                import mindspore as ms
                ms.ops.zeros(1)
                test_name = item.name
                print(f"\n[hook] pytest_runtest_teardown:{test_name}")
                peak_mem = ms.runtime.max_memory_reserved()
                ms.runtime.reset_max_memory_reserved()
                ms.runtime.empty_cache()
                fspath = item.fspath
                case_root_path = str(fspath).split("/tests/", maxsplit=1)[0]
                case_path = "tests/" + (os.path.dirname(item.fspath)).split("/tests/")[-1]
                cmd = f"echo {case_path}/{item.nodeid},max_mem:{byte_to_mb(peak_mem)} >>\
                        {case_root_path}/mem_peak.txt"
                print(f"\n{case_path}/{item.nodeid},max_mem:{byte_to_mb(peak_mem)}M")
                os.system(cmd)
            except Exception as e:
                pass
