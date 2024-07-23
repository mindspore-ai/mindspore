import pytest
import mindspore as ms
from .test_folder import import_net, import_net2
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_import(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with import code like 'from .. import module_py'.
    Expectation: Success.
    """
    import_net.run_net_with_import(mode)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_import2(mode):
    """
    Feature: Test Rewrite.
    Description: Test Rewrite with import code like 'from .. import module_cls'.
    Expectation: Success.
    """
    import_net2.run_net_with_import(mode)
