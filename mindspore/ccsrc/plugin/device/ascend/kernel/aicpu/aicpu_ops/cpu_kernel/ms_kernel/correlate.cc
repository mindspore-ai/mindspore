/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "correlate.h"
#include <cstdint>
#include <string.h>
#include "Eigen/Dense"
#include "context/inc/cpu_kernel_utils.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kCorrelate = "Correlate";
}  // namespace

namespace aicpu {
template <typename T>
void CorrelateCpuKernel::CorrelatePad(T *source_array, T *padded_array, int64_t padded_array_size, int64_t long_size,
                                      int64_t short_size, std::string mode) {
  for (int64_t i = 0; i < padded_array_size; i++) {
    padded_array[i] = static_cast<T>(0);
  }
  int start = 0;
  if (mode == "full") {
    start = short_size - 1;
  } else if (mode == "same") {
    start = short_size / 2;
  }
  for (int64_t i = 0; i < long_size; i++) {
    padded_array[start + i] = source_array[i];
  }
}

uint32_t CorrelateCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *output = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  auto a_type = ctx.Input(0)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (a_type) {
    case DT_FLOAT:
      ret = CorrelateCompute<float, float>(ctx);
      break;
    case DT_DOUBLE:
      ret = CorrelateCompute<double, double>(ctx);
      break;
    case DT_FLOAT16:
      ret = CorrelateCompute<Eigen::half, Eigen::half>(ctx);
      break;
    case DT_INT8:
      ret = CorrelateCompute<int8_t, float>(ctx);
      break;
    case DT_INT16:
      ret = CorrelateCompute<int16_t, float>(ctx);
      break;
    case DT_INT32:
      ret = CorrelateCompute<int32_t, float>(ctx);
      break;
    case DT_INT64:
      ret = CorrelateCompute<int64_t, double>(ctx);
      break;
    case DT_COMPLEX64:
      ret = CorrelateComputeComplex<std::complex<float>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = CorrelateComputeComplex<std::complex<double>>(ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "[%s] Data type of input is not support, input data type is [%s].",
                            ctx.GetOpType().c_str(), DTypeStr(a_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

template <typename T_in, typename T_out>
uint32_t CorrelateCpuKernel::CorrelateCompute(CpuKernelContext &ctx) {
  T_in *a_array = reinterpret_cast<T_in *>(ctx.Input(0)->GetData());
  T_in *v_array = reinterpret_cast<T_in *>(ctx.Input(1)->GetData());
  T_out *out_array = reinterpret_cast<T_out *>(ctx.Output(0)->GetData());
  CUST_KERNEL_CHECK_NULLPTR(ctx, a_array, KERNEL_STATUS_PARAM_INVALID, "[Correlate] Get input a failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, v_array, KERNEL_STATUS_PARAM_INVALID, "[Correlate] Get input v failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, out_array, KERNEL_STATUS_PARAM_INVALID, "[Correlate] Get output failed.");

  AttrValue *attr = ctx.GetAttr("mode");
  std::string mode = attr->GetString();
  int64_t a_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes().size();
  int64_t v_dims = ctx.Input(1)->GetTensorShape()->GetDimSizes().size();
  if (a_dims != 1 || v_dims != 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "the dimension of 'a' and 'v' should be 1-D, but got 'a' at %ld-D and 'u' at %ld-D",
                          a_dims, v_dims);
    return KERNEL_STATUS_INNER_ERROR;
  }
  int64_t a_size = ctx.Input(0)->GetTensorShape()->GetDimSizes()[0];
  int64_t v_size = ctx.Input(1)->GetTensorShape()->GetDimSizes()[0];
  if (a_size == 0 || v_size == 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "input 'a' and 'v' should not be empty, but got 'a' at (%ld) and 'u' at (%ld)", a_size,
                          v_size);
    return KERNEL_STATUS_INNER_ERROR;
  }
  bool a_ge_v = a_size >= v_size;
  int64_t long_size = v_size;
  int64_t short_size = a_size;
  if (a_ge_v) {
    long_size = a_size;
    short_size = v_size;
  }

  // step0: cast input dtype to output dtype
  T_out *casted_a_array = static_cast<T_out *>(malloc(sizeof(T_out) * a_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_a_array, KERNEL_STATUS_PARAM_INVALID,
                            "[Correlate] Malloc memory [casted_a_array] failed!")
  T_out *casted_v_array = static_cast<T_out *>(malloc(sizeof(T_out) * v_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, casted_v_array, KERNEL_STATUS_PARAM_INVALID,
                            "[Correlate] Malloc memory [casted_v_array] failed!")
  for (int64_t i = 0; i < a_size; i++) {
    casted_a_array[i] = static_cast<T_out>(a_array[i]);
  }
  for (int64_t i = 0; i < v_size; i++) {
    casted_v_array[i] = static_cast<T_out>(v_array[i]);
  }

  // step1: get padded array witch depend on mode
  int64_t out_size = long_size - short_size + 1;
  int64_t padded_long_size = long_size;
  if (mode == "same") {
    padded_long_size = long_size + short_size - 1;
    out_size = long_size;
  } else if (mode == "full") {
    padded_long_size = long_size + 2 * (short_size - 1);
    out_size = long_size + short_size - 1;
  }
  T_out *long_array = static_cast<T_out *>(malloc(sizeof(T_out) * padded_long_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, long_array, KERNEL_STATUS_PARAM_INVALID,
                            "[Correlate] Malloc memory [long_array] failed!")

  T_out *short_array;
  if (a_ge_v) {
    short_array = casted_v_array;
    CorrelatePad<T_out>(casted_a_array, long_array, padded_long_size, long_size, short_size, mode);
  } else {
    short_array = casted_a_array;
    CorrelatePad<T_out>(casted_v_array, long_array, padded_long_size, long_size, short_size, mode);
  }

  uint32_t minCoreNum = 1;
  int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  if (maxCoreNum > out_size) {
    maxCoreNum = out_size;
  }

  auto shardConv = [&long_array, &short_array, &out_array, short_size](int64_t start, int64_t end) {
    for (int64_t out_id = start; out_id < end; ++out_id) {
      T_out sum_temp = static_cast<T_out>(0);
      for (int64_t dot_id = 0; dot_id < short_size; dot_id++) {
        sum_temp += long_array[out_id + dot_id] * short_array[dot_id];
      }
      out_array[out_id] = sum_temp;
    }
  };
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, out_size, out_size / maxCoreNum, shardConv),
                           "Correlate Compute failed.");
  // step3: if a is shorter than v, then we should reverse the result
  if (!a_ge_v) {
    for (int i = 0; i < out_size / 2; i++) {
      std::swap(out_array[i], out_array[out_size - 1 - i]);
    }
  }

  free(long_array);
  free(casted_a_array);
  free(casted_v_array);
  long_array = nullptr;
  short_array = nullptr;
  casted_a_array = nullptr;
  casted_v_array = nullptr;
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t CorrelateCpuKernel::CorrelateComputeComplex(CpuKernelContext &ctx) {
  T *a_array = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  T *v_array = reinterpret_cast<T *>(ctx.Input(1)->GetData());
  T *out_array = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  CUST_KERNEL_CHECK_NULLPTR(ctx, a_array, KERNEL_STATUS_PARAM_INVALID, "[Correlate] Get input a failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, v_array, KERNEL_STATUS_PARAM_INVALID, "[Correlate] Get input v failed.");
  CUST_KERNEL_CHECK_NULLPTR(ctx, out_array, KERNEL_STATUS_PARAM_INVALID, "[Correlate] Get output failed.");

  AttrValue *attr = ctx.GetAttr("mode");
  std::string mode = attr->GetString();
  int64_t a_size = ctx.Input(0)->GetTensorShape()->GetDimSizes()[0];
  int64_t v_size = ctx.Input(1)->GetTensorShape()->GetDimSizes()[0];
  bool a_ge_v = a_size >= v_size;
  int64_t long_size = v_size;
  int64_t short_size = a_size;
  if (a_ge_v) {
    long_size = a_size;
    short_size = v_size;
  }

  // step0: get conjugate v
  T *conj_v_array = static_cast<T *>(malloc(sizeof(T) * v_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, conj_v_array, KERNEL_STATUS_PARAM_INVALID,
                            "[Correlate] Malloc memory [conj_v_array] failed!")
  for (int64_t i = 0; i < v_size; i++) {
    conj_v_array[i] = static_cast<T>(std::conj(v_array[i]));
  }

  // step1: get padded array witch depend on mode
  int64_t out_size = long_size - short_size + 1;
  int64_t padded_long_size = long_size;
  if (mode == "same") {
    padded_long_size = long_size + short_size - 1;
    out_size = long_size;
  } else if (mode == "full") {
    padded_long_size = long_size + 2 * (short_size - 1);
    out_size = long_size + short_size - 1;
  }
  T *long_array = static_cast<T *>(malloc(sizeof(T) * padded_long_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, long_array, KERNEL_STATUS_PARAM_INVALID,
                            "[Correlate] Malloc memory [long_array] failed!")
  T *short_array;
  if (a_ge_v) {
    short_array = conj_v_array;
    CorrelatePad<T>(a_array, long_array, padded_long_size, long_size, short_size, mode);
  } else {
    short_array = a_array;
    CorrelatePad<T>(conj_v_array, long_array, padded_long_size, long_size, short_size, mode);
  }

  uint32_t minCoreNum = 1;
  int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
  if (maxCoreNum > out_size) {
    maxCoreNum = out_size;
  }

  auto shardConv = [&long_array, &short_array, &out_array, short_size](int64_t start, int64_t end) {
    for (int64_t out_id = start; out_id < end; ++out_id) {
      T sum_temp = static_cast<T>(0);
      for (int64_t dot_id = 0; dot_id < short_size; dot_id++) {
        sum_temp += long_array[out_id + dot_id] * short_array[dot_id];
      }
      out_array[out_id] = sum_temp;
    }
  };
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, out_size, out_size / maxCoreNum, shardConv),
                           "Correlate Compute failed.");
  // step3: if a is shorter than v, then we should reverse the result
  if (a_ge_v == false) {
    for (int i = 0; i < out_size / 2; i++) {
      std::swap(out_array[i], out_array[out_size - 1 - i]);
    }
  }

  free(long_array);
  free(conj_v_array);
  long_array = nullptr;
  short_array = nullptr;
  conj_v_array = nullptr;
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kCorrelate, CorrelateCpuKernel);
}  // namespace aicpu
