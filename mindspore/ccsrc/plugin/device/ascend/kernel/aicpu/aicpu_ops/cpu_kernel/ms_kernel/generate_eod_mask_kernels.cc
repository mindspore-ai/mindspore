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
constexpr auto kOutputSize = 1;
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
  AttrValue *n_pos_value = ctx.GetAttr("n_pos");
  AttrValue *n_step_value = ctx.GetAttr("n_step");
  int64_t eod_token_id = (eod_token_value == nullptr) ? 0 : eod_token_value->GetInt(); // whcih bit of the element
  int64_t n_pos = (n_pos_value == nullptr) ? 0 : n_pos_value->GetInt(); // which element of tensor
  int64_t n_step = (n_step_value == nullptr) ? 0 : n_step_value->GetInt();
  switch (data_type_in) {
    case DT_FLOAT16:
      return ComputeKernel<Eigen::half, uint16_t>(ctx, n_pos, eod_token_id, n_step);
    case DT_FLOAT:
      return ComputeKernel<float, uint32_t>(ctx, n_pos, eod_token_id, n_step);
    case DT_UINT16:
      return ComputeKernel<uint16_t, uint16_t>(ctx, n_pos, eod_token_id, n_step);
    case DT_UINT32:
      return ComputeKernel<uint32_t, uint32_t>(ctx, n_pos, eod_token_id, n_step);
    case DT_UINT64:
      return ComputeKernel<uint64_t, uint64_t>(ctx, n_pos, eod_token_id, n_step);
    case DT_INT32:
      return ComputeKernel<int32_t, uint32_t>(ctx, n_pos, eod_token_id, n_step);
    case DT_INT64:
      return ComputeKernel<int64_t, uint64_t>(ctx, n_pos, eod_token_id, n_step);
    default:
      KERNEL_LOG_ERROR("GenerateEodMask kernel data type [%s] not support.", DTypeStr(data_type_in).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T, typename M>
uint32_t GenerateEodMaskCpuKernel::ComputeKernel(CpuKernelContext &ctx, const int64_t &n_pos, const int64_t &eod_token_id, const int64_t &n_step) {
  auto input_idsptr = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto input_positionptr = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  size_t batch_size = ctx.Input(0)->GetTensorShape()->GetDimSize(0);
  size_t seq_length = ctx.Input(0)->GetTensorShape()->GetDimSize(1);
  auto shard_generate_tril = [&](size_t start, size_t end) {
    for (auto i = start; i < end; ++i) {
      for (size_t j = 0; j < seq_length; ++j) {
        input_positionptr[i * seq_length + j] = input_idsptr[i * seq_length + j];
      }
    }
  };

  ++compute_count;
  auto get_per_unit_size = [&](int64_t data_size) -> int64_t {
    const int64_t max_core_num =
      std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    return data_size / std::min(max_core_num, data_size);
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch_size, get_per_unit_size(batch_size), shard_generate_tril),
                      "GenerateEodMask kernel compute failed.");
  if (compute_count == n_step || n_step == -1) {
    auto new_ds = reinterpret_cast<M*>(&input_positionptr[n_pos]);
    *new_ds = (*new_ds) ^ (1<< eod_token_id);
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kGenerateEodMask, GenerateEodMaskCpuKernel);
}  // namespace aicpu
