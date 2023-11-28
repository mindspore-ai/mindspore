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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/random_choice_with_mask_kernels.h"
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <functional>
#include "aicpu_sharder/aicpu_sharder.h"
#include "proto/aicpu_tensor.pb.h"
#include "common/distinct_uniform_int_distribution.h"
#include "common/tensor.h"
#include "common/random_utils.h"
#include "cpu_kernel/common/status.h"

namespace aicpu {
namespace {
const uint32_t kCountsIndex = 1;
const uint32_t kStatesIndex = 2;
const size_t kIndex0 = 0;
const size_t kIndex3 = 3;
const size_t kIndex4 = 4;
}  // namespace

std::vector<int64_t> GetSamples(const bool *input, int64_t input_size, int64_t count_target) {
  std::vector<int64_t> sample_ids{};
  int64_t count{0};
  for (int64_t i = 0; i < input_size; ++i) {
    if (input[i]) {
      sample_ids.push_back(i);
      count++;
    }
    if (count >= count_target) {
      break;
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

uint32_t RandomChoiceWithMaskKernel::DoCompute() {
  auto *input = reinterpret_cast<bool *>(io_addrs_[kIndex0]);
  auto *output_coordinate = reinterpret_cast<int32_t *>(io_addrs_[kIndex3]);
  auto *mask = reinterpret_cast<bool *>(io_addrs_[kIndex4]);

  int64_t input_size = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int64_t>());
  std::vector<int64_t> sample_ids = GetSamples(input, input_size, count_target_);
  size_t count = sample_ids.size();

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(sample_ids.begin(), sample_ids.end(), g);

  // Calculate coordinates
  auto *output_offset = output_coordinate;
  size_t input_rank = dims_.size();
  for (size_t i = 0; i < count; ++i) {
    CalcCoordAndWrite(sample_ids[i], output_offset, dims_);
    output_offset += input_rank;
    mask[i] = true;
  }
  return kAicpuKernelStateSucess;
}

uint32_t RandomChoiceWithMaskKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  count_target_ = attrs["count"].i();
  AICPU_LOGI("This op attr count is %d", count_target_);

  if ((count_target_ == 0) && (!unknow_shape_)) {
    AICPU_LOGE("This op attr count is 0, but the shapetype is %d", unknow_shape_);
    return kAicpuKernelStateInvalid;
  }

  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  aicpuops::TensorShape input_shape = input_tensor.tensor_shape();
  for (int j = 0; j < input_shape.dim_size(); j++) {
    dims_.push_back(input_shape.dim(j).size());
  }

  // set output to all 0's
  auto *output_coordinate = reinterpret_cast<int32_t *>(io_addrs_[kIndex3]);
  auto *mask = reinterpret_cast<bool *>(io_addrs_[kIndex4]);
  size_t input_rank = dims_.size();
  size_t output_coord_size = input_rank * count_target_ * sizeof(int32_t);
  size_t mask_size = count_target_ * sizeof(bool);
  memset_s(output_coordinate, output_coord_size, 0, output_coord_size);
  memset_s(mask, mask_size, 0, mask_size);

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t RandomChoiceWithMask(void *param) {
  aicpu::RandomChoiceWithMaskKernel random_choice_with_mask_kernel;
  return random_choice_with_mask_kernel.Compute(param);
}
}
