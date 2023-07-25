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

#include "plugin/device/gpu/kernel/nn/fast_gelu_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/fast_gelu.h"

namespace mindspore {
namespace kernel {
constexpr auto kFastGeLU = "FastGeLU";
constexpr const size_t kFastGeluInputsNum = 1;
constexpr const size_t kFastGeluOutputsNum = 1;

template <typename T>
bool FastGeLUGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kFastGeluInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kFastGeluOutputsNum, kernel_name_);
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);

  auto status = FastGelu(static_cast<size_t>(input_elements_), input_addr, output_addr,
                         reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, FastGeLUGpuKernelMod::FastGeLULaunchFunc>> FastGeLUGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &FastGeLUGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &FastGeLUGpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> FastGeLUGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FastGeLULaunchFunc> &pair) { return pair.first; });
  return support_list;
}

bool FastGeLUGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FastGeLU>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);

  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kFastGeluInputsNum || outputs.size() != kFastGeluOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kFastGeluInputsNum << " and "
                  << kFastGeluOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto pair = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!pair.first) {
    MS_LOG(ERROR) << "'" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
    return false;
  }

  kernel_func_ = func_list_[pair.second].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  return true;
}

int FastGeLUGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  std::vector<int64_t> input_shape = inputs[0]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[0]->GetShapeVector();
  auto in_shape_size = input_shape.size();
  if (in_shape_size > max_dims_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input should be less than or equal to max_dims 7, but got "
                      << in_shape_size << ".";
    return KRET_RESIZE_FAILED;
  }
  auto output_shape_size = output_shape.size();
  if (in_shape_size != output_shape_size) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input shape size should be the same as output shape size, but got"
                  << " input shape size " << in_shape_size << " output shape size" << output_shape_size;
    return KRET_RESIZE_FAILED;
  }
  // A Code Block For setting input and output shape.
  {
    input_shape_ = std::vector<size_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                       inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
    input_elements_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());
    is_null_input_ = (input_elements_ == 0);
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, FastGeLU,
                                 []() { return std::make_shared<FastGeLUGpuKernelMod>(kFastGeLU); });
}  // namespace kernel
}  // namespace mindspore
