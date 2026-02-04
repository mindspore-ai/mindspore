import os
from tests.mark_utils import arg_mark

def check_strings_in_file(file_path, strings_list):
    if not os.path.exists(file_path):
        raise ValueError("in dryrun test, the file of {file_path} not exist")
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
        for string in strings_list:
            if string not in file_content:
                return False
        return True


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=0.01)
def test_not_use_dryrunmock_not_in_simulation():
    """
    Feature: dryrun
    Description: normal run
    Expectation: no err.
    """
    cmd = "export GLOG_v=2 && python dryrun_case.py --test_case 0 >dryrun0.log 2>&1"
    ret = os.system(cmd)
    assert ret == 0
    warning_logs = ["dryrun"]
    assert not check_strings_in_file("dryrun0.log", warning_logs)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_not_use_dryrunmock_in_simulation():
    """
    Feature: dryrun
    Description: no mock in dryrun
    Expectation: catch warning logs.
    """
    cmd = "export GLOG_v=2 && python dryrun_case.py --test_case 1 >dryrun1.log 2>&1"
    ret = os.system(cmd)
    assert ret == 0
    warning_logs = ["test_ret.append(a.asnumpy())", "test_ret.append(b.asnumpy())", "if a[0][0] > 0", "if b[0, 0] > 0",
                    "b.is_contiguous()", "int(c) > 50", "float(c) > 50", "a.tolist()", "a.flush_from_cache()"]
    assert check_strings_in_file("dryrun1.log", warning_logs)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_use_dryrunmock_in_simulation():
    """
    Feature: dryrun
    Description: mock in dry run
    Expectation: result is mock val.
    """
    cmd = "export GLOG_v=2 && python dryrun_case.py --test_case 2 >dryrun3.log 2>&1"
    ret = os.system(cmd)
    assert ret == 0
    warning_logs = ["dryrun"]
    assert not check_strings_in_file("dryrun3.log", warning_logs)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_use_dryrunmock_not_in_simulation():
    """
    Feature: dryrun
    Description: mock in real run
    Expectation: result is normal.
    """
    cmd = "export GLOG_v=2 && python dryrun_case.py --test_case 3 >dryrun2.log 2>&1"
    ret = os.system(cmd)
    assert ret == 0
    warning_logs = ["dryrun"]
    assert not check_strings_in_file("dryrun2.log", warning_logs)
