import os
import platform
import pytest


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia():
    """
    Feature: test custom op of julia cases
    Description: run julia_cases
    Expectation: res == 0
    """
    system = platform.system()
    machine = platform.machine()
    if system != 'Linux' or machine != 'x86_64':
        return
    res = os.system('sh julia_run.sh')
    if res != 0:
        assert False, 'julia test fail'
