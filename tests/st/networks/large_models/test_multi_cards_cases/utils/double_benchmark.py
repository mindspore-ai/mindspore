# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Double benchmark tool.
"""
import json
import os
from typing import Union, Tuple, Optional, List

import ml_dtypes
import numpy as np


def read_np_file(file_path: str, dtype: str):
    if dtype == "bfloat16":
        array = np.fromfile(file_path, dtype=ml_dtypes.bfloat16)
        array = array.astype(np.float32)
    else:
        array = np.fromfile(file_path, dtype=dtype)
    return array


def get_err_detail(err_mask: np.ndarray,
                   npu: np.ndarray,
                   golden: np.ndarray,
                   gpu: np.ndarray = None,
                   limit: int = 10) -> str:
    """return error message detail"""
    err_positions = np.argwhere(err_mask).flatten()[:limit]
    if err_positions.size == 0:
        return ""
    err_msg = f"error elements info(first {len(err_positions)}):"
    for idx in err_positions:
        position = int(idx)
        line = f"\n{'=' * 8}>index: {position}, npu: {npu[position]}, golden: {golden[position]}"
        if gpu is not None:
            line += f", gpu: {gpu[position]}"
        err_msg += line
    return err_msg


def cal_ratio(x, y):
    return None if y == 0 else x / y


def cal_rmse(actual: np.ndarray, golden: np.ndarray) -> np.ndarray:
    """Root Mean Squared Error"""
    diff = actual - golden
    return np.sqrt(np.mean(np.square(diff)))


def cal_eb(actual: np.ndarray, golden: np.ndarray) -> np.ndarray:
    """Error Bound"""
    tensor_max = np.clip(np.abs(golden), a_min=1, a_max=None)
    diff = actual - golden
    return np.mean(diff / tensor_max)


def cal_cv(npu_max_re, gpu_max_re, err_thd):
    if npu_max_re is None or gpu_max_re is None:
        return None
    return abs(npu_max_re) / max(abs(gpu_max_re), err_thd)


def tensor_to_value(obj):
    """
    convert tensor to value.
    """
    for attr, attr_value in obj.__dict__.items():
        if isinstance(attr_value, np.ndarray) and attr_value.size == 1:
            setattr(obj, attr, attr_value.item())
        if isinstance(attr_value, (np.int32, np.int64, np.float16, np.float32, np.float64)):
            setattr(obj, attr, attr_value.item())


class REHistogramBins:
    """RE Histogram Bins"""
    ULP_HISTOGRAM_BIN = [-np.inf, -10, -5, -1, 0, 1, 5, 10, np.inf]
    ULP_HISTOGRAM_BIN_TITLES = ["Less than\n-10", "-10~-5", "-5~-1", "-1~0", "0~1", "1~5", "5~10", "Greater than\n10"]

    @classmethod
    def get_histogram_bins(cls, dtype: str) -> List[float]:
        if dtype == "float16":
            return [0, 0.001, 0.002, 0.005, 0.01, np.inf]
        if dtype == "float32":
            return [0, 0.000001, 0.00001, 0.0001, 0.0005, np.inf]
        return [0, 0.004, 0.008, np.inf]  # for bf16

    @classmethod
    def get_histogram_bin_titles(cls, dtype: str) -> List[str]:
        if dtype in (np.float16, 'float16'):
            return ["Per thousand\n0~1", "Per thousand\n1~2", "Per thousand\n2~5",
                    "Per thousand\n5~10", "Per thousand\n>10"]
        if dtype in (np.float32, 'float32'):
            return ["Per million\n0~1", "Per million\n1~10", "Per hundred thousand\n1~10",
                    "Per ten thousand\n1~10", "Per ten thousand\n>5"]
        return ["Per thousand\n0~4", "Per thousand\n4~8", "Per thousand\n>8"]  # for bf16


class PrecisionResult:
    """PrecisionResult"""
    def __init__(self):
        self.result: str = "pass"
        self.cost_time: float = 0
        self.total_cnt: Optional[int] = None
        self.total_err_cnt: Optional[int] = None
        self.normal_value_cnt: Optional[int] = None
        self.normal_err_ratio: Optional[float] = None
        self.pass_ratio: Optional[float] = None
        self.err_msg: str = ""
        self.inf_nan_cnt: Optional[int] = None
        self.inf_nan_err_cnt: Optional[int] = None
        self.finite_cnt: Optional[int] = None
        self.max_re_cv: Optional[float] = None
        self.avg_re_cv: Optional[float] = None
        self.rmse_cv: Optional[float] = None
        self.eb_cv: Optional[float] = None
        self.small_err_ratio_cv: Optional[float] = None
        self.npu_eb: Optional[float] = None
        self.npu_rmse: Optional[float] = None
        self.npu_small_info: Optional[SmallValueInfo] = None
        self.npu_normal_info: Optional[NormalValueInfo] = None
        self.gpu_rmse: Optional[float] = None
        self.gpu_eb: Optional[float] = None
        self.gpu_small_info: Optional[SmallValueInfo] = None
        self.gpu_normal_info: Optional[NormalValueInfo] = None

    def __str__(self):
        tensor_to_value(self)
        show_dict = {k: v for k, v in self.__dict__.items() if v is not None}
        return json.dumps(show_dict, indent=4, ensure_ascii=False, default=lambda obj: obj.__dict__)

    def tensor_to_value(self):
        tensor_to_value(self)


class DoubleBenchmarkStandard:
    """A class for defining benchmark standards for floating-point precision comparison.

    Attributes:
        dtype (str): The data type being benchmarked (one of "float32", "float16", "bfloat16").
        golden_dtype (str): The reference data type used for comparison.
        small_value_thd (float): Threshold below which values are considered "small".
        err_thd (float): Threshold for error comparison.
        abs_err_thd (float): Threshold for absolute error comparison.
        max_re_cv (float): Maximum relative error coefficient of variation threshold.
        avg_re_cv (float): Average relative error coefficient of variation threshold.
        rmse_cv (float): Root mean square error coefficient of variation threshold.
        small_err_ratio_cv (float): Small error ratio coefficient of variation threshold.
        eb_cv (float): Error balance coefficient of variation threshold.
    """
    def __init__(self,
                 dtype="float32",
                 small_value_thd=None,
                 err_thd=None,
                 abs_err_thd=None,
                 max_re_cv=10,
                 avg_re_cv=2,
                 rmse_cv=2,
                 small_err_ratio_cv=2,
                 eb_cv=2,
                 ):
        assert dtype in ("float32", "float16", "bfloat16")
        self.dtype = dtype
        self.golden_dtype = "float64" if self.dtype == "float32" else "float32"

        small_value_map = {
            "float32": 2 ** -20,
            "float16": 2 ** -10,
            "bfloat16": 2 ** -10,
        }
        ae_thd_map = {
            "float32": 2 ** -30,
            "float16": 1e-16,
            "bfloat16": 1e-16,
        }
        err_thd_map = {
            "float32": 2 ** -14,
            "float16": 2 ** -11,
            "bfloat16": 2 ** -8,
        }

        # small value threshold
        self.small_value_thd = small_value_map[self.dtype] if small_value_thd is None else small_value_thd
        # err threshold
        self.err_thd = err_thd_map[self.dtype] if err_thd is None else err_thd
        # abs error threshold
        self.abs_err_thd = ae_thd_map[self.dtype] if abs_err_thd is None else abs_err_thd
        # cv threshold
        self.max_re_cv = max_re_cv
        self.avg_re_cv = avg_re_cv
        self.rmse_cv = rmse_cv
        self.small_err_ratio_cv = small_err_ratio_cv
        self.eb_cv = eb_cv


class SmallValueInfo:
    """Calculate error for small values."""
    def __init__(self, actual: np.ndarray, golden: np.ndarray, finite_mask: np.ndarray, standard):
        # calculate abs error
        ae = np.abs(actual - golden)
        abs_golden = np.abs(golden)
        # small value mask
        small_value_mask = np.less(abs_golden, standard.small_value_thd)
        small_value_mask = np.logical_and(small_value_mask, finite_mask)
        # small value error mask
        small_err_mask = np.logical_and(
            np.greater(ae, standard.abs_err_thd),
            small_value_mask
        )
        self.count = np.sum(small_value_mask)
        self.err_cnt = np.sum(small_err_mask)
        self.error_ratio = cal_ratio(self.err_cnt, self.count)
        self.total_ratio = cal_ratio(self.count, golden.size)
        self.err_detail = get_err_detail(small_err_mask, actual, golden)

        if self.count == 0:
            self.max_ae_idx = None
            self.max_ae = None
            self.max_ae_actual_value = None
            self.max_ae_golden_value = None
        else:
            masked_ae = np.where(small_value_mask, ae, -1)
            self.max_ae_idx = np.argmax(masked_ae)
            self.max_ae = ae.flat[self.max_ae_idx]
            self.max_ae_actual_value = actual.flat[self.max_ae_idx]
            self.max_ae_golden_value = golden.flat[self.max_ae_idx]
        tensor_to_value(self)


class NormalValueInfo:
    """Calculate error for normal values using numpy arrays."""
    def __init__(self, actual: np.ndarray, golden: np.ndarray, finite_mask: np.ndarray, standard):
        # Calculate relative error with clipping
        abs_golden = np.abs(golden)
        denominator = np.clip(abs_golden, a_min=standard.small_value_thd, a_max=None)
        re = np.abs(actual - golden) / denominator

        # Normal value mask
        normal_value_mask = np.greater_equal(abs_golden, standard.small_value_thd)
        normal_value_mask = np.logical_and(normal_value_mask, finite_mask)

        masked_re = np.where(normal_value_mask, re, -1.0)

        self.count = np.sum(normal_value_mask)
        self.bins = REHistogramBins.get_histogram_bins(standard.dtype)
        hist, _ = np.histogram(masked_re[masked_re >= 0], bins=np.array(self.bins))

        if self.count == 0:
            self.max_re_idx = None
            self.max_re = None
            self.max_re_actual_value = None
            self.max_re_golden_value = None
            self.avg_re = None
            self.hist_ratio = [None] * (len(self.bins)-1)
        else:
            self.max_re_idx = np.argmax(masked_re)
            self.max_re = re.flat[self.max_re_idx]
            self.max_re_actual_value = actual.flat[self.max_re_idx]
            self.max_re_golden_value = golden.flat[self.max_re_idx]
            self.avg_re = np.mean(re[normal_value_mask])
            self.hist_ratio = [f"{(x/self.count*100):.2f}%" if self.count != 0 else "N/A" for x in hist]
        tensor_to_value(self)


class DoubleBenchmarkComparator:
    """Comparator with double benchmark."""

    @classmethod
    def format_to_np(
            cls,
            standard: DoubleBenchmarkStandard,
            golden: Union[np.ndarray, str],
            npu: Union[np.ndarray, str],
            gpu: Union[np.ndarray, str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Format data to np.ndarray."""
        if isinstance(golden, str):
            if os.path.exists(golden):
                golden = read_np_file(golden, standard.golden_dtype)
            else:
                raise FileNotFoundError(f"not found golden data:{golden}")

        dtype = standard.dtype
        # format npu to torch tensor
        if isinstance(npu, str):
            if os.path.exists(npu):
                npu = read_np_file(npu, dtype)
            else:
                raise FileNotFoundError(f"not found npu data:{npu}")

        # format gpu to torch tensor
        if isinstance(gpu, str):
            if os.path.exists(gpu):
                gpu = read_np_file(gpu, dtype)
            else:
                raise FileNotFoundError(f"not found gpu data:{gpu}")

        # flatten
        golden = golden.flatten()
        npu = npu.flatten()
        gpu = gpu.flatten() if gpu is not None else None
        return golden, npu, gpu

    @classmethod
    def check_inf_nan(cls, npu_data, gpu_data, golden_data, result):
        """Compare inf or nan value using numpy arrays."""
        # nan error
        npu_nan_mask = np.isnan(npu_data)
        gpu_nan_mask = np.isnan(gpu_data)
        golden_nan_mask = np.isnan(golden_data)
        nan_mask = npu_nan_mask | gpu_nan_mask | golden_nan_mask
        nan_err_mask = (npu_nan_mask != golden_nan_mask) & (npu_nan_mask != gpu_nan_mask)

        # inf error
        npu_inf_mask = np.isinf(npu_data)
        gpu_inf_mask = np.isinf(gpu_data)
        golden_inf_mask = np.isinf(golden_data)
        inf_mask = npu_inf_mask | gpu_inf_mask | golden_inf_mask
        inf_err_mask = (npu_data != golden_data) & (npu_data != gpu_data) & inf_mask

        # inf/nan error
        inf_nan_mask = inf_mask | nan_mask
        finite_mask = ~inf_nan_mask
        inf_nan_err_mask = nan_err_mask | inf_err_mask

        result.inf_nan_cnt = np.sum(inf_nan_mask)
        result.finite_cnt = np.sum(finite_mask)
        result.inf_nan_err_cnt = np.sum(inf_nan_err_mask)

        err_msg = ""
        if result.inf_nan_err_cnt != 0:
            err_detail = get_err_detail(inf_nan_err_mask, npu_data, golden_data, gpu_data)
            err_msg = f"inf/nan mismatch with golden/gpu, err_cnt={result.inf_nan_err_cnt}\n{err_detail}"

        return finite_mask, err_msg

    @classmethod
    def check_threshold(cls, actual, thd, desc, result, num=None):
        """Check if error value is over threshold or not."""
        if actual is not None and actual > thd:
            err_msg = f"{desc} exceeds threshold: {actual} > thd: {thd}"
            if desc in ["Error balance", "Error balance CV", "Error ratio", "Large range error ratio"]:
                if num >= 1e6:
                    result.result = "error"
                    result.err_msg += f"{err_msg}\n"
                else:
                    err_msg = f"Insufficient test samples ({num}) < 1M, {err_msg}"
                    result.err_msg += f"{err_msg}\n"
            else:
                result.result = "error"
                result.err_msg += f"{err_msg}\n"

    @classmethod
    def apply(cls, npu_data, gpu_data, golden_data, standard=DoubleBenchmarkStandard()):
        """Calculate double benchmark."""
        result = PrecisionResult()
        golden_data, npu_data, gpu_data = cls.format_to_np(standard, golden_data, npu_data, gpu_data)
        result.total_cnt = golden_data.size

        # check inf/nan
        finite_mask, err_msg = cls.check_inf_nan(npu_data, gpu_data, golden_data, result)
        if result.inf_nan_err_cnt != 0:
            result.result = "error"
            result.err_msg += f"{err_msg}\n"

        # check small value ae
        npu_small_info = SmallValueInfo(npu_data, golden_data, finite_mask, standard)
        gpu_small_info = SmallValueInfo(gpu_data, golden_data, finite_mask, standard)
        result.npu_small_info = npu_small_info
        result.gpu_small_info = gpu_small_info

        # check normal value re
        npu_normal_info = NormalValueInfo(npu_data, golden_data, finite_mask, standard)
        gpu_normal_info = NormalValueInfo(gpu_data, golden_data, finite_mask, standard)
        result.npu_normal_info = npu_normal_info
        result.gpu_normal_info = gpu_normal_info

        # check rmse
        npu_finite = npu_data[finite_mask]
        gpu_finite = gpu_data[finite_mask]
        golden_finite = golden_data[finite_mask]
        result.npu_rmse = cal_rmse(npu_finite, golden_finite)
        result.gpu_rmse = cal_rmse(gpu_finite, golden_finite)

        # check EB
        result.npu_eb = cal_eb(npu_finite, golden_finite)
        result.gpu_eb = cal_eb(gpu_finite, golden_finite)

        # do_summary
        err_thd = standard.err_thd
        result.max_re_cv = cal_cv(npu_normal_info.max_re, gpu_normal_info.max_re, err_thd)
        result.avg_re_cv = cal_cv(npu_normal_info.avg_re, gpu_normal_info.avg_re, err_thd)
        result.rmse_cv = cal_cv(result.npu_rmse, result.gpu_rmse, err_thd)
        result.eb_cv = cal_cv(result.npu_eb, result.gpu_eb, err_thd)
        result.small_err_ratio_cv = cal_cv(
            result.npu_small_info.error_ratio, result.gpu_small_info.error_ratio, err_thd
        )

        # check_threshold
        cls.check_threshold(result.eb_cv, standard.eb_cv, "Error balance CV", result, result.finite_cnt)
        cls.check_threshold(result.max_re_cv, standard.max_re_cv, "Max relative error CV", result)
        cls.check_threshold(result.avg_re_cv, standard.avg_re_cv, "Mean relative error CV", result)
        cls.check_threshold(result.rmse_cv, standard.rmse_cv, "Root mean square error CV", result)
        cls.check_threshold(result.small_err_ratio_cv, standard.small_err_ratio_cv,
                            "Small value error ratio CV", result)
        return result

    @classmethod
    def check_pass_or_not(cls, npu_data, gpu_data, golden_data, standard=DoubleBenchmarkStandard()):
        result = cls.apply(npu_data, gpu_data, golden_data, standard)
        if result.result == "pass":
            return True
        raise Exception(result.err_msg)
