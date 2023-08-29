import pytest
import mindspore as ms
from .test_folder import import_net, import_net2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_import(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with import code like 'from .. import module_py'.
    Expectation: Success.
    """
    import_net.run_net_with_import(mode)

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_import2(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with import code like 'from .. import module_cls'.
    Expectation: Success.
    """
    import_net2.run_net_with_import(mode)
