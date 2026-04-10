import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_inner_sync_batch_norm_forward_world_size_2_channel_2_dim_2_msrun():
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    ret = os.system("msrun --worker_num=2 --local_worker_num=2 --master_port=8975 --log_dir=msrun_log --join=True "
                    "--cluster_time_out=100 pytest -s -v "
                    "test_inner_syncbatchnorm.py::test_inner_sync_batch_norm_forward_world_size_2_channel_2_dim_2")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_inner_sync_batch_norm_forward_world_size_2_channel_3_dim_4_diff_nhw_msrun():
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    ret = os.system("msrun --worker_num=2 --local_worker_num=2 --master_port=8975 --log_dir=msrun_log --join=True "
                    "--cluster_time_out=100 pytest -s -v "
                    "test_inner_syncbatchnorm.py::test_inner_sync_batch_norm_forward_world_size_2_channel_3_dim_4_diff_nhw")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_inner_sync_batch_norm_backward_world_size_2_channel_2_dim_2_msrun():
    """
    Feature: Ops.
    Description: test op sync_batch_norm.
    Expectation: expect correct result.
    """
    ret = os.system("msrun --worker_num=2 --local_worker_num=2 --master_port=8975 --log_dir=msrun_log --join=True "
                    "--cluster_time_out=100 pytest -sv --disable-warnings "
                    "test_inner_syncbatchnorm.py::test_inner_sync_batch_norm_backward_world_size_2_channel_2_dim_2")
    assert ret == 0
