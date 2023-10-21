/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "cpu_kernel/ms_kernel/argmin_with_value.h"
#include <vector>
#include <string>
#include <algorithm>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kArgMinWithValueInputNum = 1;
const uint32_t kArgMinWithValueOutputNum = 2;
const int64_t kParallelDivision = 128 * 1024;
const char *kArgMinWithValue = "ArgMinWithValue";
}  // namespace

namespace aicpu {
template <class T>
uint32_t ExecArgMinWithValue(const CpuKernelContext &ctx) {
  // Get Tensors
  Tensor *input_tensor = ctx.Input(kFirstInputIndex);
  Tensor *indice_tensor = ctx.Output(kFirstOutputIndex);
  Tensor *values_tensor = ctx.Output(kSecondOutputIndex);
  // Get raw ptrs
  KERNEL_CHECK_NULLPTR(input_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.")
  KERNEL_CHECK_NULLPTR(indice_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.")
  KERNEL_CHECK_NULLPTR(values_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 1 data failed.")
  const T *input = reinterpret_cast<T *>(input_tensor->GetData());
  int32_t *indice = reinterpret_cast<int32_t *>(indice_tensor->GetData());
  T *values = reinterpret_cast<T *>(values_tensor->GetData());
  // Process attrs
  auto input_shape = input_tensor->GetTensorShape()->GetDimSizes();
  int64_t input_shape_size = static_cast<int64_t>(input_shape.size());
  KERNEL_CHECK_NULLPTR(ctx.GetAttr("dimension"), KERNEL_STATUS_PARAM_INVALID, "Get attr 'dimension' data failed.")
  int64_t dim = ctx.GetAttr("dimension")->GetInt();
  int64_t upper_bound_included = static_cast<int64_t>(input_shape_size - 1);
  int64_t lower_bound_included = static_cast<int64_t>(-input_shape_size);
  if (dim > upper_bound_included || dim < lower_bound_included) {
    if (input_shape_size == 0) {
      if (dim != -1 && dim != 0) {
        KERNEL_LOG_ERROR("[ArgMinWithValue] Dimension is out of range.");
        return KERNEL_STATUS_PARAM_INVALID;
      }
    } else {
      KERNEL_LOG_ERROR("[ArgMinWithValue] Dimension is out of range.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (input_shape_size == 0) {
    indice[0] = 0;
    values[0] = input[0];
    return KERNEL_STATUS_OK;
  }
  dim = dim < 0 ? input_shape_size + dim : dim;
  int64_t num_outer = 1;
  int64_t num_inner = 1;
  for (int64_t i = 0; i < input_shape_size; i++) {
    if (i < dim) {
      num_outer *= input_shape[i];
    } else if (i > dim) {
      num_inner *= input_shape[i];
    }
  }
  int64_t dim_on = input_shape[dim];

  // Core computation
  auto argmin_wv_shard = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; i++) {
      int64_t outer_index = i * dim_on * num_inner;
      for (int64_t j = 0; j < num_inner; ++j) {
        std::vector<T> values_vec;
        int64_t inner_index = outer_index + j;
        for (int64_t k = 0; k < dim_on; k++) {
          int64_t index_on = k * num_inner + inner_index;
          values_vec.push_back(input[index_on]);
        }
        int32_t indice_on = std::distance(values_vec.begin(), std::min_element(values_vec.begin(), values_vec.end()));
        int32_t dst_index = i * num_inner + j;
        indice[dst_index] = indice_on;
        int32_t src_index = indice_on * num_inner + inner_index;
        values[dst_index] = input[src_index];
      }
    }
  };

  int64_t data_num = ctx.Input(kFirstInputIndex)->NumElements() * static_cast<int64_t>(sizeof(T));
  if (data_num <= kParallelDivision) {
    argmin_wv_shard(0, num_outer);
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    if (max_core_num > num_outer) {
      max_core_num = num_outer;
    }
    // log error if max_core num is 0
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("[ArgMinWithValue] max_core_num is 0.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, num_outer, num_outer / max_core_num, argmin_wv_shard),
                        "[ArgMinWithValue] Compute failed.");
  }
  return KERNEL_STATUS_OK;
}

uint32_t ArgMinWithValueCpuKernel::Compute(CpuKernelContext &ctx) {
  const std::vector<std::string> required_attrs = {"dimension"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kArgMinWithValueInputNum, kArgMinWithValueOutputNum, required_attrs),
                      "[ArgMinWithValue] Check input_num, output_num, required_attr failed.");
  auto data_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (data_type) {
    case DT_FLOAT16:
      return ExecArgMinWithValue<Eigen::half>(ctx);
      break;
    case DT_FLOAT:
      return ExecArgMinWithValue<float>(ctx);
      break;
    case DT_DOUBLE:
      return ExecArgMinWithValue<double>(ctx);
      break;
    case DT_INT8:
      return ExecArgMinWithValue<int8_t>(ctx);
      break;
    case DT_INT16:
      return ExecArgMinWithValue<int16_t>(ctx);
      break;
    case DT_INT32:
      return ExecArgMinWithValue<int32_t>(ctx);
      break;
    case DT_INT64:
      return ExecArgMinWithValue<int64_t>(ctx);
      break;
    case DT_UINT8:
      return ExecArgMinWithValue<uint8_t>(ctx);
      break;
    case DT_UINT16:
      return ExecArgMinWithValue<uint16_t>(ctx);
      break;
    case DT_UINT32:
      return ExecArgMinWithValue<uint32_t>(ctx);
      break;
    case DT_UINT64:
      return ExecArgMinWithValue<uint64_t>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("[ArgMinWithValue] Data type [%s] is not supported.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kArgMinWithValue, ArgMinWithValueCpuKernel);
}  // namespace aicpu
