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
#include <functional>
#include <algorithm>
#include "plugin/device/gpu/kernel/nn/dropout_gpu_kernel.h"
#include "mindspore/core/ops/dropout.h"
#include "kernel/philox_random.h"

namespace mindspore {
namespace kernel {
constexpr size_t kDropoutInputNum = 1;
constexpr size_t kDropoutOutputNum = 2;

bool DropoutFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Dropout>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast Dropout ops failed!";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDropoutInputNum, kernel_ptr->name());
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDropoutOutputNum, kernel_ptr->name());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_ptr->name()
                      << "', it does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  uint64_t seed0 = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("Seed0")));
  uint64_t seed1 = static_cast<uint64_t>(GetValue<int64_t>(base_operator->GetAttr("Seed1")));
  seed_ = random::GetSeed(seed0, seed1);
  return true;
}

int DropoutFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs,
                                   const std::map<uint32_t, tensor::TensorPtr> &others) {
  ResetResource();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Dropout>(base_operator);
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  if (!(CHECK_SHAPE_POSITIVE(input_shape_))) {
    is_null_input_ = true;
    InitSizeLists();
    return 0;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(!input_shape_.empty(), "input shape should not be empty!");
  num_count_ = std::accumulate(input_shape_.begin(), input_shape_.end(), 1, std::multiplies<size_t>());
  input_size_ = abstract::TypeIdSize(inputs[kIndex0]->GetDtype()) * num_count_;
  output_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype()) * num_count_;
  InitSizeLists();
  keep_prob_ = kernel_ptr->get_keep_prob();
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  num_count_ = std::accumulate(input_shape_.begin(), input_shape_.end(), int64_t(1), std::multiplies<int64_t>());
  if (num_count_ % kDropoutTileSize == 0) {
    use_fused_dropout_ = true;
    if (base_operator->HasAttr(kAttrOnlyUseFirstOutput)) {
      only_use_first_output_ = GetValue<bool>(base_operator->GetAttr(kAttrOnlyUseFirstOutput));
    } else if (base_operator->HasAttr(kAttrOnlyUseSecondOutput)) {
      only_use_second_output_ = GetValue<bool>(base_operator->GetAttr(kAttrOnlyUseSecondOutput));
    }
  }
  if (!states_init_ && !use_fused_dropout_) {
    CHECK_CURAND_RET_WITH_EXCEPT(curandCreateGenerator(&mask_generator_, CURAND_RNG_PSEUDO_DEFAULT),
                                 "Failed to create generator");
    states_init_ = true;
  }
  return 0;
}

void DropoutFwdGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  num_count_ = 0;
  keep_prob_ = 0.0;
  use_fused_dropout_ = false;
  only_use_first_output_ = false;
  only_use_second_output_ = false;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

void DropoutFwdGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_size_);
  if (only_use_second_output_) {
    output_size_list_.push_back(1);
  } else {
    output_size_list_.push_back(input_size_);  // output size: the same with input size
  }
  if (only_use_first_output_) {
    output_size_list_.push_back(1);
  } else {
    output_size_list_.push_back(input_size_);  // mask size: the same with input size
  }
  if (!use_fused_dropout_) {
    workspace_size_list_.push_back(num_count_ * sizeof(float));  // temp mask_f for curandGen
  }
}

template <typename T>
bool DropoutFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                          const std::vector<AddressPtr> &workspace,
                                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }

  T *input = GetDeviceAddress<T>(inputs, 0);
  T *output = GetDeviceAddress<T>(outputs, 0);
  T *mask = GetDeviceAddress<T>(outputs, 1);

  cudaError_t status = cudaErrorNotReady;
  if (use_fused_dropout_) {
    if (only_use_first_output_) {
      status = FusedDropoutForwardOnlyOutput(input, output, num_count_, keep_prob_, seed_, seed_offset_,
                                             reinterpret_cast<cudaStream_t>(stream_ptr));
    } else if (only_use_second_output_) {
      status = FusedDropoutForwardOnlyMask(mask, num_count_, keep_prob_, seed_, seed_offset_,
                                           reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      status = FusedDropoutForward(input, mask, output, num_count_, keep_prob_, seed_, seed_offset_,
                                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    CHECK_CUDA_STATUS(status, kernel_name_);
    seed_offset_ += num_count_;
    return true;
  }

  float *mask_f = GetDeviceAddress<float>(workspace, 0);

  CHECK_CURAND_RET_WITH_EXCEPT(curandSetPseudoRandomGeneratorSeed(mask_generator_, seed_ + seed_offset_),
                               "Failed to SetPseudoRandomGeneratorSeed");
  CHECK_CURAND_RET_WITH_EXCEPT(curandSetStream(mask_generator_, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "Failed to set stream for generator");
  // curandGen only support float or double for mask.
  CHECK_CURAND_RET_WITH_EXCEPT(curandGenerateUniform(mask_generator_, mask_f, num_count_),
                               "Failed to generate uniform");
  status =
    DropoutForward(input, mask, output, mask_f, num_count_, keep_prob_, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  seed_offset_ += 1;

  return true;
}

std::vector<std::pair<KernelAttr, DropoutFwdGpuKernelMod::DropoutFunc>> DropoutFwdGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &DropoutFwdGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &DropoutFwdGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &DropoutFwdGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> DropoutFwdGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DropoutFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Dropout, DropoutFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
