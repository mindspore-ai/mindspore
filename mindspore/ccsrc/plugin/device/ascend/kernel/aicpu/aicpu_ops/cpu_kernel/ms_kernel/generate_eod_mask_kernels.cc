/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "./generate_eod_mask_kernels.h"
#include <Eigen/Dense>
#include <map>
#include <thread>
#include <numeric>
#include <vector>
#include <functional>
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const char *kGenerateEodMask = "GenerateEodMask";
constexpr auto kInputSize = 1;
constexpr auto kOutputSize = 2;
constexpr auto kInputIdsShape = 2;
constexpr auto kAddressSize = 3;
constexpr auto kDim0 = 0;
constexpr auto kDim1 = 1;
constexpr auto kDim2 = 2;
constexpr auto kDim3 = 3;
}  // namespace
namespace aicpu {
uint32_t GenerateEodMaskCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputSize, kOutputSize), "GenerateEodMaskCpu check input and output failed.");
  Tensor *input = ctx.Input(0);
  auto data_type_in = input->GetDataType();
  AttrValue *eod_token_value = ctx.GetAttr("eod_token_id");
  int64_t eod_token_id = (eod_token_value == nullptr) ? 0 : eod_token_value->GetInt();
  switch (data_type_in) {
    case DT_UINT16:
      return ComputeKernel<uint16_t>(ctx, eod_token_id);
    case DT_UINT32:
      return ComputeKernel<uint32_t>(ctx, eod_token_id);
    case DT_UINT64:
      return ComputeKernel<uint64_t>(ctx, eod_token_id);
    case DT_INT32:
      return ComputeKernel<int32_t>(ctx, eod_token_id);
    case DT_INT64:
      return ComputeKernel<int64_t>(ctx, eod_token_id);
    default:
      KERNEL_LOG_ERROR("GenerateEodMask kernel data type [%s] not support.", DTypeStr(data_type_in).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t GenerateEodMaskCpuKernel::ComputeKernel(CpuKernelContext &ctx, const T &eod_token_id) {
  auto input_idsptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_positionptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  auto outputptr = reinterpret_cast<Eigen::half *>(ctx.Output(1)->GetData());
  auto output_shape = ctx.Output(1)->GetTensorShape()->GetDimSizes();
  size_t output_size =
    std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int64_t>()) * sizeof(Eigen::half);
  if (memset_s(outputptr, output_size, 0x0, output_size) != EOK) {
    KERNEL_LOG_ERROR("memset_s failed!");
    return KERNEL_STATUS_INNER_ERROR;
  }
  size_t batch_size = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  size_t seq_length = ctx.Input(0)->GetTensorShape()->GetDimSize(1);

  auto shard_generate_tril = [&](size_t start, size_t end) {
    size_t x = seq_length * seq_length;
    for (size_t i = start; i < end; ++i) {
      for (size_t j = 0; j < seq_length; ++j) {
        for (size_t k = 0; k < j + 1; ++k) {
          outputptr[i * x + j * seq_length + k] = (Eigen::half)1.0;
        }
      }
    }
  };

  auto shard_generate_eod_mask = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      T sub = 0;
      T pre_sub = 0;
      for (size_t index = 0; index < seq_length; ++index) {
        size_t sub_index = i * seq_length + index;
        if (input_idsptr[sub_index] == eod_token_id) {
          pre_sub = sub;
          sub = index + 1;
          size_t seq_n2 = seq_length * seq_length;
          for (size_t k = index + 1; k < seq_length; ++k) {
            for (size_t m = 0; m < index + 1; ++m) {
              outputptr[i * seq_n2 + k * seq_length + m] = (Eigen::half)0.0;
            }
          }
          input_positionptr[sub_index] = index - pre_sub;
        } else {
          input_positionptr[sub_index] = index - sub;
        }
      }
    }
  };

  auto get_per_unit_size = [&](int64_t data_size) -> int64_t {
    const int64_t max_core_num =
      std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    return data_size / std::min(max_core_num, data_size);
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch_size, get_per_unit_size(batch_size), shard_generate_tril),
                      "GenerateEodMask kernel compute failed.");
  KERNEL_HANDLE_ERROR(
    CpuKernelUtils::ParallelFor(ctx, batch_size, get_per_unit_size(batch_size), shard_generate_eod_mask),
    "GenerateEodMask kernel compute failed.");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kGenerateEodMask, GenerateEodMaskCpuKernel);
}  // namespace aicpu
