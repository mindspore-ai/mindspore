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

#include "kernel/cpu/native/random_shuffle_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <complex>
#include "mindspore/ops/infer/random_shuffle.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace random_shuffle_cpu {
namespace {
constexpr size_t kRandomShuffleInputsNum = 1;
constexpr size_t kRandomShuffleOutputsNum = 1;
constexpr size_t kScalarShapeSize = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool RandomShuffleCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed")));
  uint64_t seed2 = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr("seed2")));
  uint64_t init_seed = random::GetSeed(seed, seed2);
  rng_.seed(init_seed);
  batch_rank_ = LongToSize(ops::get_batch_rank(primitive_));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "RandomShuffle does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RandomShuffleCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = NativeCpuKernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
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

  return ret;
}

template <typename T>
bool RandomShuffleCpuKernelMod::ScalarShuffle(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<kernel::KernelTensor *> &outputs,
                                              const std::vector<size_t> &perm) const {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  for (size_t i = 0; i < perm.size(); i++) {
    output[i] = input[perm[i]];
  }
  return true;
}

template <typename T>
bool RandomShuffleCpuKernelMod::ScalarShuffleWithBatchRank(const std::vector<kernel::KernelTensor *> &inputs,
                                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  auto task = [this, input, output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      std::vector<size_t> perm(shuffle_size_);
      int n = 0;
      std::generate(perm.begin(), perm.end(), [&n] { return n++; });
      std::shuffle(perm.begin(), perm.end(), rng_);
      size_t offset = i * shuffle_size_ * inner_size_;
      for (size_t j = 0; j < perm.size(); j++) {
        output[offset + j] = input[offset + perm[j]];
      }
    }
  };
  ParallelLaunchAutoSearch(task, outer_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool RandomShuffleCpuKernelMod::TensorShuffle(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<kernel::KernelTensor *> &outputs,
                                              const std::vector<size_t> &perm) {
  auto input = GetDeviceAddress<int8_t>(inputs, kIndex0);
  auto output = GetDeviceAddress<int8_t>(outputs, kIndex0);
  auto task = [this, output, input, perm](size_t start, size_t end) {
    size_t copy_size = inner_size_ * sizeof(T);
    for (size_t i = start; i < end; i++) {
      size_t output_offset = i * copy_size;
      size_t input_offset = perm[i] * copy_size;
      auto ret = memcpy_s(output + output_offset, copy_size, input + input_offset, copy_size);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed, ret=" << ret;
      }
    }
  };
  ParallelLaunchAutoSearch(task, shuffle_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool RandomShuffleCpuKernelMod::TensorShuffleWithBatchRank(const std::vector<kernel::KernelTensor *> &inputs,
                                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<int8_t>(inputs, kIndex0);
  auto output = GetDeviceAddress<int8_t>(outputs, kIndex0);
  auto outer_task = [this, output, input](size_t outer_start, size_t outer_end) {
    size_t copy_size = inner_size_ * sizeof(T);
    for (size_t k = outer_start; k < outer_end; k++) {
      std::vector<size_t> perm(shuffle_size_);
      int n = 0;
      std::generate(perm.begin(), perm.end(), [&n] { return n++; });
      std::shuffle(perm.begin(), perm.end(), rng_);
      size_t offset = k * shuffle_size_ * copy_size;
      for (size_t i = 0; i < shuffle_size_; i++) {
        size_t output_offset = offset + i * copy_size;
        size_t input_offset = offset + perm[i] * copy_size;
        auto ret = memcpy_s(output + output_offset, copy_size, input + input_offset, copy_size);
        if (ret != EOK) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed, ret=" << ret;
        }
      }
    }
  };
  ParallelLaunchAutoSearch(outer_task, outer_size_, this, &parallel_search_info_);
  return true;
}

template <typename T>
bool RandomShuffleCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRandomShuffleInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRandomShuffleOutputsNum, kernel_name_);

  if (input_shape_.empty() || input_shape_[batch_rank_] <= 1) {
    auto ret = memcpy_s(outputs[0]->device_ptr(), outputs[0]->size(), inputs[0]->device_ptr(), inputs[0]->size());
    if (ret != EOK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', memcpy failed, ret=" << ret;
    }
    return ret == EOK;
  }

  shuffle_size_ = LongToSize(input_shape_[batch_rank_]);
  std::vector<size_t> perm(shuffle_size_);
  int n = 0;
  std::generate(perm.begin(), perm.end(), [&n] { return n++; });
  std::shuffle(perm.begin(), perm.end(), rng_);

  bool ret = true;
  if (batch_rank_ == 0) {
    if (input_shape_.size() <= kScalarShapeSize) {
      ret = ScalarShuffle<T>(inputs, outputs, perm);
    } else {
      ret = TensorShuffle<T>(inputs, outputs, perm);
    }
  } else {
    if (input_shape_.size() <= batch_rank_ + kScalarShapeSize) {
      ret = ScalarShuffleWithBatchRank<T>(inputs, outputs);
    } else {
      ret = TensorShuffleWithBatchRank<T>(inputs, outputs);
    }
  }

  return ret;
}

std::vector<std::pair<KernelAttr, RandomShuffleCpuKernelMod::RandomShuffleFunc>> RandomShuffleCpuKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
    &RandomShuffleCpuKernelMod::LaunchKernel<float16>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    &RandomShuffleCpuKernelMod::LaunchKernel<float>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    &RandomShuffleCpuKernelMod::LaunchKernel<double>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
    &RandomShuffleCpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
    &RandomShuffleCpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    &RandomShuffleCpuKernelMod::LaunchKernel<int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    &RandomShuffleCpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
    &RandomShuffleCpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
    &RandomShuffleCpuKernelMod::LaunchKernel<uint16_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
    &RandomShuffleCpuKernelMod::LaunchKernel<uint32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
    &RandomShuffleCpuKernelMod::LaunchKernel<uint64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
    &RandomShuffleCpuKernelMod::LaunchKernel<bool>},
   {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
    &RandomShuffleCpuKernelMod::LaunchKernel<complex64>},
   {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
    &RandomShuffleCpuKernelMod::LaunchKernel<complex128>}};

std::vector<KernelAttr> RandomShuffleCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RandomShuffleFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RandomShuffle, RandomShuffleCpuKernelMod);
}  // namespace random_shuffle_cpu
}  // namespace kernel
}  // namespace mindspore
