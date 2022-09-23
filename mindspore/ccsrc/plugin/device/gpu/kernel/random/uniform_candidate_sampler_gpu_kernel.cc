/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/random/uniform_candidate_sampler_gpu_kernel.h"
#include <algorithm>
#include "mindspore/core/ops/uniform_candidate_sampler.h"

namespace mindspore {
namespace kernel {
bool UniformCandidateSamplerGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t input_num = 1;
  constexpr size_t output_num = 3;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;

  auto kernel_ptr = std::dynamic_pointer_cast<ops::UniformCandidateSampler>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  // getting attrs
  num_true_ = kernel_ptr->get_num_true();
  num_sampled_ = kernel_ptr->get_num_sampled();
  unique_ = kernel_ptr->get_unique();
  range_max_ = kernel_ptr->get_range_max();
  remove_accidental_hits_ = kernel_ptr->get_remove_accidental_hits();
  init_seed_ = kernel_ptr->get_seed();
  if (init_seed_ == 0) {
    cur_seed_ = time(NULL);
    generator_.seed(cur_seed_);
  } else {
    generator_.seed(init_seed_);
  }
  return true;
}

int UniformCandidateSamplerGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                const std::vector<KernelTensorPtr> &inputs,
                                                const std::vector<KernelTensorPtr> &outputs,
                                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = LongVecToSizeVec(inputs[kIndex0]->GetDeviceShapeAdaptively());
  // check the rank of input in infer shape.
  input_size_ = input_shape[0] * input_shape[1];
  if (num_sampled_ + static_cast<int64_t>(input_size_) > range_max_) {
    remove_accidental_hits_ = false;
  }
  return KRET_OK;
}

template <typename T, typename S>
bool UniformCandidateSamplerGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                       const std::vector<AddressPtr> &,
                                                       const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (init_seed_ == 0 && cur_seed_ == 0) {
    // Update current seed.
    cur_seed_ = time(NULL);
    generator_.seed(cur_seed_);
  }
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
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "UniformCandidateSampler cudaDeviceSyncFailed");
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
  CalUniformCandidateSampler(static_cast<int64_t>(input_size_), num_sampled_, value, true_expected_count,
                             sampled_expected_count, reinterpret_cast<cudaStream_t>(stream_ptr));
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
