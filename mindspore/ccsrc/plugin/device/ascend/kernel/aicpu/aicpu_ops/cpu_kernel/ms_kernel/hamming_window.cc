/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "hamming_window.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"

namespace {
const char *kHammingWindow = "HammingWindow";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
constexpr int64_t kParallelDataNums = 16 * 1024;
constexpr int64_t kParallelDataNumsMid = 7 * 1024;

#define WINDOW_LENGTH_CASE(DTYPE, TYPE, LENGTH, CTX)                       \
  case (DTYPE): {                                                          \
    TYPE *length_addr = reinterpret_cast<TYPE *>(ctx.Input(0)->GetData()); \
    LENGTH = static_cast<int64_t>(*length_addr);                           \
    break;                                                                 \
  }

#define SWITCH_PARALLEL(SHARD, end_num)                                                           \
  if (end_num >= kParallelDataNumsMid) {                                                          \
    uint32_t min_core_num = 1;                                                                    \
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);     \
    if (end_num < kParallelDataNums) {                                                            \
      max_core_num = std::min(max_core_num, 4L);                                                  \
    }                                                                                             \
    if (max_core_num > end_num) {                                                                 \
      max_core_num = end_num;                                                                     \
    }                                                                                             \
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, end_num, end_num / max_core_num, SHARD), \
                        "HammingWindow #SHARD Compute failed.");                                  \
  } else {                                                                                        \
    SHARD(0, end_num);                                                                            \
  }
}  // namespace

namespace aicpu {
uint32_t HammingWindowCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "HammingWindow check input and output number failed.");
  int64_t dtype = 0;
  AttrValue *dtype_attr = ctx.GetAttr("dtype");
  if (dtype_attr != nullptr) {
    dtype = dtype_attr->GetInt();
  }
  DataType data_type = static_cast<DataType>(dtype);
  ctx.Output(0)->SetDataType(data_type);
  switch (data_type) {
    case DT_FLOAT:
      return HammingWindowCompute<float>(ctx);
    case DT_FLOAT16:
      return HammingWindowCompute<Eigen::half>(ctx);
    case DT_DOUBLE:
      return HammingWindowCompute<double>(ctx);
    default:
      KERNEL_LOG_ERROR(
        "Attribute dtype only supports floating point types, "
        "but got:[%s].",
        DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t HammingWindowCpuKernel::HammingWindowCompute(CpuKernelContext &ctx) {
  DataType input_type = ctx.Input(0)->GetDataType();
  int64_t length;
  switch (input_type) {
    WINDOW_LENGTH_CASE(DT_INT8, int8_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_INT16, int16_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_INT32, int32_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_INT64, int64_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_UINT8, uint8_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_UINT16, uint16_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_UINT32, uint32_t, length, ctx)
    WINDOW_LENGTH_CASE(DT_UINT64, uint64_t, length, ctx)
    default:
      KERNEL_LOG_ERROR("HammingWindow input data type [%s] not support.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((length >= 0), KERNEL_STATUS_PARAM_INVALID,
                     "Input window length cannot be negative, bug got [%d].", length);

  Tensor *y_tensor = ctx.Output(0);
  auto y_shape = y_tensor->GetTensorShape();
  std::vector<int64_t> y_dim = y_shape->GetDimSizes();
  y_dim.clear();
  if (length != 0) {
    y_dim.push_back(length);
  }
  y_shape->SetDimSizes(y_dim);
  y_tensor->SetTensorShape(y_shape.get());
  y_tensor->SetDataSize(length * sizeof(T));
  T *y_addr = reinterpret_cast<T *>(y_tensor->GetData());

  if (length == 0) {
    return KERNEL_STATUS_OK;
  } else if (length == 1) {
    *y_addr = T{1};
    return KERNEL_STATUS_OK;
  } else {
    bool periodic = true;
    AttrValue *periodic_attr = ctx.GetAttr("periodic");
    if (periodic_attr != nullptr) {
      periodic = periodic_attr->GetBool();
    }
    int64_t window_length = length;
    if (periodic) {
      length += 1;
    }
    float alpha = 0.54;
    AttrValue *alpha_attr = ctx.GetAttr("alpha");
    if (alpha_attr != nullptr) {
      alpha = alpha_attr->GetFloat();
    }
    float beta = 0.46;
    AttrValue *beta_attr = ctx.GetAttr("beta");
    if (beta_attr != nullptr) {
      beta = beta_attr->GetFloat();
    }
    constexpr double t_pi = 6.283185307179586476925286766559;
    auto shard_hamming_window = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        double result = alpha - beta * std::cos(i * t_pi / (length - 1));
        *(y_addr + i) = static_cast<T>(result);
      }
    };
    SWITCH_PARALLEL(shard_hamming_window, window_length);
    return KERNEL_STATUS_OK;
  }
}

REGISTER_CPU_KERNEL(kHammingWindow, HammingWindowCpuKernel);
}  // namespace aicpu
