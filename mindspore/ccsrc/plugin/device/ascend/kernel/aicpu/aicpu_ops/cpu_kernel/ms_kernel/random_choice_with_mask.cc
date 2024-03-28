/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "ms_kernel/random_choice_with_mask.h"
#include <random>
#include <complex>
#include <algorithm>
#include <securec.h>
#include "context/inc/cpu_kernel_utils.h"
#include "random/utils.h"

namespace {
const char *kRandomChoiceWithMask = "RandomChoiceWithMask";
constexpr size_t kRandomChoiceWithMaskInputNum = 1;
constexpr size_t kRandomChoiceWithMaskOutputNum = 2;
}  // namespace

namespace aicpu {
namespace {
std::vector<int64_t> GetAllSamples(const bool *input, int64_t input_size) {
  std::vector<int64_t> sample_ids{};
  for (int64_t i = 0; i < input_size; ++i) {
    if (input[i]) {
      sample_ids.push_back(i);
    }
  }
  return sample_ids;
}

void CalcCoordAndWrite(int64_t idx, int32_t *output, const std::vector<int64_t> &dims) {
  for (int i = dims.size() - 1; i >= 0; --i) {
    output[i] = idx % dims[i];
    idx /= dims[i];
  }
}
}  // namespace

uint32_t RandomChoiceWithMaskCpuKernel::RandomChoiceWithMaskCompute(CpuKernelContext &ctx) {
  auto input = reinterpret_cast<bool *>(ctx.Input(0)->GetData());
  auto output_coordinate = reinterpret_cast<int32_t *>(ctx.Output(0)->GetData());
  auto mask = reinterpret_cast<bool *>(ctx.Output(1)->GetData());

  int64_t input_size = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int64_t>());
  std::vector<int64_t> sample_ids = GetAllSamples(input, input_size);

  uint64_t seed = std::random_device()();
  std::mt19937 g(seed);
  std::shuffle(sample_ids.begin(), sample_ids.end(), g);
  size_t count = std::min(sample_ids.size(), static_cast<size_t>(count_target_));

  // Calculate coordinates
  auto *output_offset = output_coordinate;
  size_t input_rank = dims_.size();
  for (size_t i = 0; i < count; ++i) {
    CalcCoordAndWrite(sample_ids[i], output_offset, dims_);
    output_offset += input_rank;
    mask[i] = true;
  }
  return KERNEL_STATUS_OK;
}

uint32_t RandomChoiceWithMaskCpuKernel::Compute(CpuKernelContext &ctx) {
  NormalCheck(ctx, kRandomChoiceWithMaskInputNum, kRandomChoiceWithMaskOutputNum);
  auto input = ctx.Input(0);
  dims_ = input->GetTensorShape()->GetDimSizes();

  // set output to all 0's
  auto *output_data = reinterpret_cast<int32_t *>(ctx.Output(0)->GetData());
  auto *mask = reinterpret_cast<bool *>(ctx.Output(1)->GetData());
  size_t input_rank = dims_.size();
  auto count_ptr = ctx.GetAttr("count");
  CUST_KERNEL_CHECK_NULLPTR(ctx, count_ptr, KERNEL_STATUS_INNER_ERROR, "Failed to get attr 'count'.");
  count_target_ = count_ptr->GetInt();
  size_t output_coord_size = input_rank * count_target_ * sizeof(int32_t);
  size_t mask_size = count_target_ * sizeof(bool);
  auto ret = memset_s(output_data, output_coord_size, 0, output_coord_size);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_INNER_ERROR, "memset failed.");
  ret = memset_s(mask, mask_size, 0, mask_size);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_INNER_ERROR, "memset failed.");

  return RandomChoiceWithMaskCompute(ctx);
}

REGISTER_MS_CPU_KERNEL(kRandomChoiceWithMask, RandomChoiceWithMaskCpuKernel);
}  // namespace aicpu
