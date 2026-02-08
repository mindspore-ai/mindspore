/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/random/random_shuffle_gpu_kernel.h"
#include <functional>
#include <utility>
#include <memory>
#include <string>
#include <algorithm>
#include <complex>
#include "ir/anf.h"
#include "utils/log_adapter.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"
#include "include/cuda_fp16.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/random_shuffle_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

constexpr size_t kRandomShuffleInputsNum = 1;
constexpr size_t kRandomShuffleOutputsNum = 1;
constexpr size_t kScalarShapeSize = 1;
}  // namespace

bool RandomShuffleGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (primitive_->HasAttr(ops::kBatchRank)) {
    batch_rank_ = LongToSize(GetValue<int64_t>(primitive_->GetAttr(ops::kBatchRank)));
  }
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  seed_ = random::GetSeed(seed, seed2);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "RandomShuffle does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RandomShuffleGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRandomShuffleInputsNum, kernel_name_);

  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  MS_EXCEPTION_IF_NULL(inputs[0]);
  input_shape_ = inputs[0]->GetShapeVector();
  if (!input_shape_.empty() && batch_rank_ >= input_shape_.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the batch_rank should be less than input shape, but got batch_rank: " << batch_rank_
                  << ", input shape: " << input_shape_;
    return KRET_RESIZE_FAILED;
  }

  outer_size_ = 1;
  for (size_t i = 0; i < batch_rank_; i++) {
    outer_size_ *= input_shape_[i];
  }
  inner_size_ = 1;
  for (size_t j = batch_rank_ + 1; j < input_shape_.size(); j++) {
    inner_size_ *= input_shape_[j];
  }

  if (input_shape_.size() > batch_rank_) {
    shuffle_size_ = LongToSize(input_shape_[batch_rank_]);
  } else {
    shuffle_size_ = 1;
  }

  workspace_size_list_.push_back(sizeof(int) * shuffle_size_ * outer_size_);
  return ret;
}

std::vector<int> RandomShuffleGpuKernelMod::GetShuffleIndex() {
  generator_.seed(seed_ + seed_offset_);
  std::vector<int> perm(shuffle_size_);
  int n = 0;
  std::generate(perm.begin(), perm.end(), [&n] { return n++; });
  std::shuffle(perm.begin(), perm.end(), generator_);
  return perm;
}

template <typename T>
bool RandomShuffleGpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRandomShuffleInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRandomShuffleOutputsNum, kernel_name_);

  auto *input_addr = GetDeviceAddress<T>(inputs, 0);
  auto *workspace_addr = GetDeviceAddress<int>(workspace, 0);
  auto *output_addr = GetDeviceAddress<T>(outputs, 0);
  if (input_shape_.empty() || input_shape_[batch_rank_] <= 1) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_addr, input_addr, inputs[0]->size(), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "RandomShuffle cudaMemcpy failed.");
    return true;
  }

  if (input_shape_.size() <= batch_rank_ + kScalarShapeSize) {
    for (int64_t i = 0; i < outer_size_; i++) {
      std::vector<int> perm = GetShuffleIndex();
      size_t offset = i * inner_size_ * SizeToLong(shuffle_size_);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(workspace_addr + i * shuffle_size_, perm.data(), shuffle_size_ * sizeof(int),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
        "RandomShuffle cudaMemcpy failed.");
      auto status = ScalarShuffle(SizeToLong(shuffle_size_), workspace_addr, input_addr + offset, output_addr + offset,
                                  device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
  } else {
    for (int64_t i = 0; i < outer_size_; i++) {
      std::vector<int> perm = GetShuffleIndex();
      size_t offset = i * inner_size_ * SizeToLong(shuffle_size_);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(workspace_addr + i * shuffle_size_, perm.data(), shuffle_size_ * sizeof(int),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream_)),
        "RandomShuffle cudaMemcpy failed.");
      auto status = TensorShuffle(SizeToLong(shuffle_size_), inner_size_, workspace_addr, input_addr + offset,
                                  output_addr + offset, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
  }
  seed_offset_ += 1;

  return true;
}

std::vector<std::pair<KernelAttr, RandomShuffleGpuKernelMod::RandomShuffleFunc>> RandomShuffleGpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    &RandomShuffleGpuKernelMod::LaunchKernel<half>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &RandomShuffleGpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    &RandomShuffleGpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    &RandomShuffleGpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    &RandomShuffleGpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    &RandomShuffleGpuKernelMod::LaunchKernel<int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    &RandomShuffleGpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    &RandomShuffleGpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    &RandomShuffleGpuKernelMod::LaunchKernel<uint16_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    &RandomShuffleGpuKernelMod::LaunchKernel<uint32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    &RandomShuffleGpuKernelMod::LaunchKernel<uint64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    &RandomShuffleGpuKernelMod::LaunchKernel<bool>},
   {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
    &RandomShuffleGpuKernelMod::LaunchKernel<Complex<float>>},
   {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
    &RandomShuffleGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> RandomShuffleGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RandomShuffleFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, RandomShuffle, RandomShuffleGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
