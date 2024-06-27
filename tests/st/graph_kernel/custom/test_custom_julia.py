import os
import platform
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
