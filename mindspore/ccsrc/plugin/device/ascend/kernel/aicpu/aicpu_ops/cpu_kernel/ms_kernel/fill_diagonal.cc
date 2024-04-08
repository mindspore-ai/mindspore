/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include "fill_diagonal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <set>
#include "Eigen/Dense"

#include "context/inc/cpu_kernel_utils.h"
#include "log.h"
#include "context/common/status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "securec.h"

namespace {
const uint32_t kInputNum = 1;
const uint32_t kOutputNum = 1;
const uint32_t InputDimLimit = 2;
const std::vector<std::string> attr_names{"fill_value"};
const char *const kFillDiagonal = "FillDiagonal";
constexpr int64_t kParallelDataNums = 512 * 1024;

#define FILLDIAGONAL_COMPUTE_CASE(DTYPE, TYPE, INPUT_DIMS, STRIDE, HEIGHT, WIDTH, CTX) \
  case (DTYPE): {                                                                      \
    uint32_t result = FillDiag<TYPE>(INPUT_DIMS, STRIDE, HEIGHT, WIDTH, CTX);          \
    if (result != KERNEL_STATUS_OK) {                                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "FillDiagonal kernel compute failed.");               \
      return result;                                                                   \
    }                                                                                  \
    break;                                                                             \
  }
}  // namespace

namespace aicpu {
inline bool IsUnsignedType(DataType dataType) {
  static const std::set<DataType> unsigned_types{DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64};
  return unsigned_types.count(dataType) > 0;
}

uint32_t FillDiagonalCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum, attr_names),
                           "FillDiagonal check input and output number failed or "
                           "attr[fill_value] is nullptr.");

  Tensor *input = ctx.Input(0);
  auto input_shape = input->GetTensorShape();
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_shape, KERNEL_STATUS_PARAM_INVALID, "FillDiagonal Get input shape failed.")
  int64_t input_dims = input_shape->GetDims();
  CUST_KERNEL_CHECK_FALSE(ctx, input_dims >= InputDimLimit, KERNEL_STATUS_PARAM_INVALID,
                          "FillDiagonal input dims must larger than 1.");
  DataType input_dtype = input->GetDataType();
  AttrValue *fill_value_attr = ctx.GetAttr("fill_value");
  fill_value_ = fill_value_attr->GetFloat();
  if (IsUnsignedType(input_dtype) && fill_value_ < 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "For FillDiagonal, [fill_value] should be non-negative for input of unsigned type.");
    return KERNEL_STATUS_INNER_ERROR;
  }

  int64_t height = input_shape->GetDimSize(0);
  int64_t width = input_shape->GetDimSize(1);

  if (input_dims > InputDimLimit) {
    int64_t h_dim = height;
    for (int64_t i = 1; i < input_dims; i++) {
      CUST_KERNEL_CHECK_FALSE(ctx, input_shape->GetDimSize(i) == h_dim, KERNEL_STATUS_PARAM_INVALID,
                              "FillDiagonal each dim of input must be of "
                              "equal length while dims > 2.");
    }
  }

  int64_t stride = 0;
  for (int64_t i = (input_dims - 1); i >= 0; i--) {
    stride += static_cast<int64_t>(std::round(std::pow(width, i)));
  }

  switch (input_dtype) {
    FILLDIAGONAL_COMPUTE_CASE(DT_FLOAT16, Eigen::half, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_FLOAT, float, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_DOUBLE, double, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_UINT8, uint8_t, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_INT8, int8_t, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_INT16, int16_t, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_INT32, int32_t, input_dims, stride, height, width, ctx)
    FILLDIAGONAL_COMPUTE_CASE(DT_INT64, int64_t, input_dims, stride, height, width, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "FillDiagonal kernel data type [%s] not support.", DTypeStr(input_dtype).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }

  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FillDiagonalCpuKernel::FillDiag(int64_t input_dims, int64_t stride, int64_t height, int64_t width,
                                         CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(0);
  T *input_data = reinterpret_cast<T *>(input->GetData());
  T *output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  uint64_t output_size = ctx.Output(0)->GetDataSize();

  AttrValue *wrap_attr = ctx.GetAttr("wrap");
  bool wrap = wrap_attr == nullptr ? false : wrap_attr->GetBool();

  int64_t data_nums = input->NumElements();

  int64_t size = std::min(height, width);
  if (data_nums <= kParallelDataNums) {
    CUST_KERNEL_CHECK_FALSE(ctx, (memcpy_s(output_data, output_size, input_data, data_nums * sizeof(T)) == EOK),
                            KERNEL_STATUS_INNER_ERROR, "FillDiagonal memcpy failed, dst len is %ld, src size is %ld.",
                            output_size, data_nums * sizeof(T));
  } else {
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > data_nums) {
      max_core_num = data_nums;
    }
    auto shard_copy = [&](size_t start, size_t end) {
      auto size = (end - start) * sizeof(T);
      auto ret = memcpy_s(output_data + start, output_size - (start * sizeof(T)), input_data + start, size);
      if (ret != EOK) {
        CUST_KERNEL_LOG_ERROR(ctx, "FillDiagonal memcpy failed, src: %p, dest: %p, size: %zu.", input_data + start,
                              output_data + start, size);
      }
    };

    uint32_t ret = KERNEL_STATUS_INNER_ERROR;
    if (max_core_num > 0) {
      ret = CpuKernelUtils::ParallelFor(ctx, data_nums, data_nums / max_core_num, shard_copy);
    }
    if (ret != KERNEL_STATUS_OK) {
      CUST_KERNEL_LOG_ERROR(ctx, "CpuKernelUtils::ParallelFor shared_copy failed.");
      return KERNEL_STATUS_INNER_ERROR;
    }
  }

  for (int64_t i = 0; i < size; ++i) {
    output_data[stride * i] = static_cast<T>(fill_value_);
  }

  if (wrap && input_dims == InputDimLimit && height > width + 1) {
    int64_t location = size * (size + 1);
    while (location < data_nums) {
      output_data[location] = static_cast<T>(fill_value_);
      location += stride;
    }
  }

  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kFillDiagonal, FillDiagonalCpuKernel);
}  // namespace aicpu