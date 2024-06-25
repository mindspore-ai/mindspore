from ..share.ops.primitive.p_range_ops import OpsRangeFactory
import pytest
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_1_limit_5_delta_2_max_50_int32():
    """
    Feature: Ops.
    Description: range算子正向测试，start=1, limit=5, delta=2,maxlen=50, int32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=1, limit=5, delta=2, maxlen=50, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_limit_100_delta_2_max_200_fp32():
    """
    Feature: Ops.
    Description: range算子正向测试，start=0.2, limit=100, delta=2, maxlen=200, float32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=0.2, limit=100, delta=2, maxlen=200, dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_320_limit_1000_delta_032_max_2500_fp32():
    """
    Feature: Ops.
    Description: range算子正向测试，start=320, limit=1000.8, delta=0.32,maxlen=2500, float32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=320, limit=1000.8, delta=0.32, maxlen=2500, dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_neg_int32():
    """
    Feature: Ops.
    Description: range算子正向测试，start=-1, int32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=-1, limit=1000, delta=1, maxlen=3500, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_neg_fp32():
    """
    Feature: Ops.
    Description: range算子正向测试，start=-1, int32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=-0.5, limit=1000.8, delta=0.32, maxlen=3500, dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_limit_neg_int32():
    """
    Feature: Ops.
    Description: range算子正向测试，limit=-1, float32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=-100, limit=-1, delta=1, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_maxlen_10000000_int32():
    """
    Feature: Ops.
    Description: range算子正向测试，maxlen=1千万, int32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=1, limit=9000000, delta=1, maxlen=10000000, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_maxlen_10000000_float32():
    """
    Feature: Ops.
    Description: range算子正向测试，maxlen=1千万, float32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=1, limit=8000, delta=0.01, maxlen=10000000, dtype=np.float32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_limit_0_int32():
    """
    Feature: Ops.
    Description: range算子正向测试，limit=0 int32.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=0, limit=0, delta=1, maxlen=3500, dtype=np.int32)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_1_limit_5_delta_2_max_50_int64():
    """
    Feature: Ops.
    Description: range算子正向测试，start=1, limit=5, delta=2,maxlen=50, int64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=1, limit=5, delta=2, maxlen=50)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_limit_100_delta_2_max_200_fp64():
    """
    Feature: Ops.
    Description: range算子正向测试，start=0.2, limit=100, delta=2, maxlen=200, float64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=0.2, limit=100, delta=2, maxlen=200, dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_320_limit_1000_delta_032_max_2500_fp64():
    """
    Feature: Ops.
    Description: range算子正向测试，start=320, limit=1000.8, delta=0.32,maxlen=2500, float64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=320, limit=1000.8, delta=0.32, maxlen=2500, dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_neg_int64():
    """
    Feature: Ops.
    Description: range算子正向测试，start=-1, int64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=-1, limit=1000, delta=1, maxlen=3500, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_start_neg_fp64():
    """
    Feature: Ops.
    Description: range算子正向测试，start=-0.5, float64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=-0.5, limit=1000.8, delta=0.32, maxlen=3500, dtype=np.float64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_limit_neg_int64():
    """
    Feature: Ops.
    Description: range算子正向测试，limit=-1, float64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=-100, limit=-1, delta=1, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_maxlen_10000000_int64():
    """
    Feature: Ops.
    Description: range算子正向测试，maxlen=1千万, int64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=1, limit=9000000, delta=1, maxlen=10000000, dtype=np.int64)
    fact.forward_cmp()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_p_range_input_maxlen_10000000_float64():
    """
    Feature: Ops.
    Description: range算子正向测试，maxlen=1千万, float64.
    Expectation: expect correct result.
    """
    fact = OpsRangeFactory(start=1, limit=8000, delta=0.01, maxlen=10000000, dtype=np.float64)
    fact.forward_cmp()
