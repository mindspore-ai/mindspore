import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia():
    """
    Feature: test custom op of julia cases
    Description: run julia_cases
    Expectation: res == 0
    """
    res = os.system('sh julia_run.sh')
    if res != 0:
        assert False, 'julia test fail'
