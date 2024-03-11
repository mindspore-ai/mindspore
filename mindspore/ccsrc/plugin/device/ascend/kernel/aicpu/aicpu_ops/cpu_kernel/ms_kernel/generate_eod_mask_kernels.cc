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
#include "./generate_eod_mask_kernels.h"
#include <Eigen/Dense>
#include <map>
#include <thread>
#include <numeric>
#include <vector>
#include <functional>
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"

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
  AttrValue *n_error_mode = ctx.GetAttr("n_error_mode");
  int64_t eod_token_id = (eod_token_value == nullptr) ? 0 : eod_token_value->GetInt();  // which bit of the element
  int64_t n_pos = (n_pos_value == nullptr) ? 0 : n_pos_value->GetInt();                 // which element of tensor
  std::vector<int64_t> n_step = (n_step_value == nullptr) ? std::vector<int64_t>{-1} : n_step_value->GetListInt();
  std::string error_mode = n_error_mode == nullptr ? "specific" : n_error_mode->GetString();
  auto pos = error_mode.find("-");
  std::string mask_nfirst = "";
  bool enable_mask_nfirst = false;
  if (pos != std::string::npos) {
    mask_nfirst = error_mode.substr(pos + 1);
    error_mode = error_mode.substr(0, pos);
  }
  if (mask_nfirst.compare("mask_nfirst") == 0) {
    enable_mask_nfirst = true;
  }

  int64_t circle = -1;

  if (error_mode.compare("circle") == 0) {
    if (n_step.size() >= 2 || n_step[0] <= 0) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
    circle = n_step[0];
  } else if (error_mode.compare("specific") != 0) {
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type_in) {
    case DT_FLOAT16:
      return ComputeKernel<Eigen::half, uint16_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_FLOAT:
      return ComputeKernel<float, uint32_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_BFLOAT16:
      return ComputeKernel<Eigen::bfloat16, uint16_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_UINT16:
      return ComputeKernel<uint16_t, uint16_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_UINT32:
      return ComputeKernel<uint32_t, uint32_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_UINT64:
      return ComputeKernel<uint64_t, uint64_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_INT32:
      return ComputeKernel<int32_t, uint32_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    case DT_INT64:
      return ComputeKernel<int64_t, uint64_t>(ctx, n_pos, eod_token_id, n_step, circle, enable_mask_nfirst);
    default:
      KERNEL_LOG_ERROR("GenerateEodMask kernel data type [%s] not support.", DTypeStr(data_type_in).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T, typename M>
uint32_t GenerateEodMaskCpuKernel::ComputeKernel(CpuKernelContext &ctx, const int64_t &n_pos,
                                                 const int64_t &eod_token_id, const std::vector<int64_t> &n_step,
                                                 const int64_t &circle, const bool &enable_mask_nfirst) {
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

  auto get_per_unit_size = [&](int64_t data_size) -> int64_t {
    const int64_t max_core_num =
      std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
    return data_size / std::min(max_core_num, data_size);
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch_size, get_per_unit_size(batch_size), shard_generate_tril),
                      "GenerateEodMask kernel compute failed.");
  for (uint32_t i = 0; i < n_step.size(); ++i) {
    bool condition = circle >= 1 && _compute_count != 0 && _compute_count % circle == 0;
    if (condition || _compute_count == n_step[i] || n_step[i] == -1) {
      // cppcheck-suppress *
      auto new_ds = reinterpret_cast<M *>(&input_positionptr[n_pos]);

      if (enable_mask_nfirst) {
        // flip the last n pos if 1
        int64_t bit_size = sizeof(*new_ds) * 8;
        auto total_length = std::min(eod_token_id + 1, bit_size - 1);
        // j=1 to skip the flag bit
        for (uint32_t j = 1; j < total_length; ++j) {
          if (((*new_ds) & (1 << (bit_size - 1 - j))) == 0) {
            (*new_ds) = (*new_ds) ^ (1 << (bit_size - 1 - j));
            break;
          }
        }
      } else {
        *new_ds = (*new_ds) ^ (1 << eod_token_id);
      }
      break;
    }
  }
  ++_compute_count;
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kGenerateEodMask, GenerateEodMaskCpuKernel);
}  // namespace aicpu
