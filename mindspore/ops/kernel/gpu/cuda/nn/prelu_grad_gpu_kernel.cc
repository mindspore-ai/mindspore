/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/nn/prelu_grad_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool PReLUGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 2;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int PReLUGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto x_shape = LongVecToSizeVec(inputs[kIndex1]->GetShapeVector());
  auto weight_shape = LongVecToSizeVec(inputs[kIndex2]->GetShapeVector());
  is_null_input_ =
    CHECK_SHAPE_NULL(x_shape, kernel_name_, "x") || CHECK_SHAPE_NULL(weight_shape, kernel_name_, "weight");
  input_length_ = std::accumulate(x_shape.begin(), x_shape.end(), size_t(1), std::multiplies<>());
  size_t x_rank = x_shape.size();
  size_t channel_num;
  if (x_rank == 0) {
    channel_num = 1;
    per_channel_length_ = 1;
  } else if (x_rank == 1) {
    channel_num = 1;
    per_channel_length_ = x_shape[0];
  } else {
    channel_num = x_shape[1];
    const size_t beg_pos = 2;
    per_channel_length_ = std::accumulate(x_shape.begin() + beg_pos, x_shape.end(), size_t(1), std::multiplies<>());
  }

  if (weight_shape.size() != 1 || (weight_shape[0] != 1 && weight_shape[0] != channel_num)) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of weight must be equal to 1 and "
                  << "weight.shape[0] must be equal to 1 or the channel number, but got the dimension of "
                  << "weight: " << weight_shape.size() << ", weight.shape[0]: " << weight_shape[0]
                  << ", the channel num: " << channel_num;
    return KRET_RESIZE_FAILED;
  }
  weight_length_ = weight_shape[0];
  workspace_size_ = weight_length_ * IntToSize(GET_BLOCKS(input_length_) * GET_THREADS) * sizeof(float);
  workspace_size_list_.push_back(workspace_size_);
  return KRET_OK;
}

template <typename T>
bool PReLUGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  auto *dy = GetDeviceAddress<T>(inputs, kIndex0);
  auto *x = GetDeviceAddress<T>(inputs, kIndex1);
  auto *w = GetDeviceAddress<T>(inputs, kIndex2);
  auto *dx = GetDeviceAddress<T>(outputs, kIndex0);
  auto *dw = GetDeviceAddress<T>(outputs, kIndex1);
  auto *dw_array = GetDeviceAddress<float>(workspace, kIndex0);

  auto status = CalPReLUGrad(input_length_, weight_length_, per_channel_length_, dy, x, w, dx, dw, dw_array,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, PReLUGradGpuKernelMod::PReLUGradLaunchFunc>> PReLUGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &PReLUGradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &PReLUGradGpuKernelMod::LaunchKernel<float>},
};

std::vector<KernelAttr> PReLUGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, PReLUGradGpuKernelMod::PReLUGradLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, PReLUGrad, PReLUGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
