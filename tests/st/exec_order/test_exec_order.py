import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_gpto_exec_order():
    """
    Feature: this test call gpto_net.py
    Description: this test use msrun to run the gpto test
    Expectation: the test should pass without any error
    """
    return_code = os.system(
        "msrun --worker_num=1 --local_worker_num=1 --master_addr=127.0.0.1 "
        "--master_port=10969 --join=True gpto_net.py"
    )

    assert return_code == 0
