import pytest
import mindspore as ms
from .test_folder import import_net


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_import(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with two father classes, one of them has class variables.
    Expectation: Success.
    """
    import_net.run_net_with_import(mode)
