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

#include "plugin/device/gpu/kernel/math/multinomial_gpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
static constexpr size_t input_num_ = 2;
static constexpr size_t output_num_ = 1;
}  // namespace
bool MultinomialGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num_, kernel_name_);

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Multinomial>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  seed_ = static_cast<int>(kernel_ptr->get_seed());
  seed2_ = static_cast<int>(kernel_ptr->get_seed2());
  auto &allocator = device::gpu::GPUMemoryAllocator::GetInstance();
  rand_state_ = static_cast<curandState *>(allocator.AllocTensorMem(sizeof(curandState) * distributions_));

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
  }
  launch_func_ = func_list_[index].second;
  return true;
}

int MultinomialGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  workspace_size_list_.clear();
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  auto input_shape_0 = Convert2SizeTClipNeg(inputs[0]->GetShapeVector());
  if (input_shape_0.size() == 1) {
    distributions_ = 1;
    categories_ = input_shape_0[0];
  } else {
    distributions_ = input_shape_0[0];
    categories_ = input_shape_0[1];
  }
  size_t elem_num = std::accumulate(input_shape_0.begin(), input_shape_0.end(), 1, std::multiplies<size_t>());

  workspace_size_list_.emplace_back(elem_num * sizeof(float));
  return ret;
}

bool MultinomialGpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num_, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num_, kernel_name_);

  launch_func_(this, inputs, outputs, stream_ptr);
  return true;
}

template <typename T>
void MultinomialGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &outputs, void *stream_ptr) {
  int *output_addr = GetDeviceAddress<int>(outputs, 0);
  T *probs_addr = GetDeviceAddress<T>(inputs, 0);
  int64_t *num_sample_addr = GetDeviceAddress<int64_t>(inputs, 1);
  if (distributions_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', divide by zero. the distributions_ is 0.";
  }

  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  if (!rand_state_init_) {
    int rng_seed = 0;
    std::random_device rd;
    if (seed2_ != 0) {
      rng_seed = seed2_;
    } else if (seed_ != 0) {
      rng_seed = seed_;
    } else {
      rng_seed = static_cast<int>(rd());
    }
    InitRandState(rng_seed, distributions_, rand_state_, stream);
    rand_state_init_ = true;
  }

  Multinomial(distributions_, categories_, probs_addr, rand_state_, num_sample_addr, output_addr, stream);
}

std::vector<std::pair<KernelAttr, MultinomialGpuKernelMod::LaunchFunc>> MultinomialGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &MultinomialGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> MultinomialGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Multinomial, MultinomialGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
