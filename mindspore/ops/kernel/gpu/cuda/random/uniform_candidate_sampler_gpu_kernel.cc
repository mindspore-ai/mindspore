/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/random/uniform_candidate_sampler_gpu_kernel.h"
#include <algorithm>
#include "mindspore/ops/infer/uniform_candidate_sampler.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"

namespace mindspore {
namespace kernel {
bool UniformCandidateSamplerGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 3;

  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  // getting attrs
  num_true_ = GetValue<int64_t>(primitive_->GetAttr(ops::kNumTrue));
  num_sampled_ = GetValue<int64_t>(primitive_->GetAttr(ops::kNumSampled));
  unique_ = GetValue<bool>(primitive_->GetAttr(ops::kUnique));
  range_max_ = GetValue<int64_t>(primitive_->GetAttr(ops::kRangeMax));
  remove_accidental_hits_ = GetValue<bool>(primitive_->GetAttr("remove_accidental_hits"));
  uint64_t seed = static_cast<uint64_t>(GetValue<int64_t>(primitive_->GetAttr(ops::kSeed)));
  uint64_t init_seed = random::GetSeed(seed, 0);
  rng_.seed(init_seed);

  return true;
}

int UniformCandidateSamplerGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetShapeVector());
  // check the rank of input in infer shape.
  input_size_ = input_shape[0] * input_shape[1];
  if (num_sampled_ + static_cast<int64_t>(input_size_) > range_max_) {
    remove_accidental_hits_ = false;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool UniformCandidateSamplerGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &,
                                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  T *sampled_candidates = GetDeviceAddress<T>(outputs, kIndex0);
  S *true_expected_count = GetDeviceAddress<S>(outputs, kIndex1);
  S *sampled_expected_count = GetDeviceAddress<S>(outputs, kIndex2);
  std::set<T> set_input;
  if (remove_accidental_hits_) {
    T *input = GetDeviceAddress<T>(inputs, kIndex0);
    auto array_input = std::vector<T>(input_size_, kIndex0);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&array_input[0], input, input_size_ * sizeof(T), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "UniformCandidateSampler cudaMemcpyAsync sampled_candidates failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "UniformCandidateSampler cudaStreamSyncFailed");
    for (const auto item : array_input) {
      set_input.insert(item);
    }
  }
  std::vector<T> sampled_candidates_device;
  int64_t counter = Sampling(set_input, &sampled_candidates_device);
  S prob = Probability<S>();
  size_t sampled_candidates_size = num_sampled_ * sizeof(T);
  S value = ApproximateExpectedCount<S>(prob, num_sampled_, counter);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(sampled_candidates, &sampled_candidates_device[0], sampled_candidates_size, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "UniformCandidateSampler cudaMemcpyAsync sampled_candidates failed");
  auto status = CalUniformCandidateSampler(static_cast<int64_t>(input_size_), num_sampled_, value, true_expected_count,
                                           sampled_expected_count, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, UniformCandidateSamplerGpuKernelMod::UCSGpuLaunchFunc>>
  UniformCandidateSamplerGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &UniformCandidateSamplerGpuKernelMod::LaunchKernel<int, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &UniformCandidateSamplerGpuKernelMod::LaunchKernel<int64_t, float>},
};

std::vector<KernelAttr> UniformCandidateSamplerGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UniformCandidateSamplerGpuKernelMod::UCSGpuLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UniformCandidateSampler, UniformCandidateSamplerGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
