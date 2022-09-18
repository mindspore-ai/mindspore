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

#include "plugin/device/gpu/kernel/nn/maxpool_with_argmax_grad_gpu_kernel.h"
#include "mindspore/core/ops/grad/max_pool_grad_with_argmax.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMaxPoolGradWithArgmaxInputsNum = 3;
constexpr size_t kMaxPoolGradWithArgmaxOutputsNum = 1;
int MaxPoolWithArgmaxGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  input_size_list_.clear();
  output_size_list_.clear();
  size_t input_num = inputs.size();
  if (input_num != kMaxPoolGradWithArgmaxInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != kMaxPoolGradWithArgmaxOutputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  auto x_shape = inputs[kIndex0]->GetShapeVector();
  auto dy_shape = inputs[kIndex1]->GetShapeVector();
  auto index_shape = inputs[kIndex2]->GetShapeVector();
  auto dx_shape = outputs[kIndex0]->GetShapeVector();
  if (!IsValidShape(x_shape) || !IsValidShape(dy_shape) || !IsValidShape(index_shape) || !IsValidShape(dx_shape)) {
    return static_cast<int>(KRET_UNKNOWN_SHAPE);
  }
  if (x_shape.size() < kXDimLowerLimit || dy_shape.size() < kDyDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of x and dy cannot be less than 4, but got "
                      << "the dimension of x: " << x_shape.size() << ", the dimension of dy: " << dy_shape.size();
  }
  x_size_ = x_type_size_ * SizeOf(x_shape);
  dy_size_ = dy_type_size_ * SizeOf(dy_shape);
  index_size_ = idx_type_size_ * SizeOf(index_shape);
  dx_size_ = dx_type_size_ * SizeOf(dx_shape);

  n_ = LongToSizeClipNeg(x_shape[kXIndexForN]);
  c_ = LongToSizeClipNeg(x_shape[kXIndexForC]);
  x_height_ = LongToSizeClipNeg(x_shape[kXIndexForH]);
  x_width_ = LongToSizeClipNeg(x_shape[kXIndexForW]);
  dy_height_ = LongToSizeClipNeg(dy_shape[kDyIndexForH]);
  dy_width_ = LongToSizeClipNeg(dy_shape[kDyIndexForW]);
  input_size_list_.push_back(dy_size_);
  input_size_list_.push_back(index_size_);
  output_size_list_.push_back(dx_size_);
  return static_cast<int>(KRET_OK);
}

bool MaxPoolWithArgmaxGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolGradWithArgmax>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast MaxPoolGradWithArgmax ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  size_t input_num = inputs.size();
  if (input_num != kMaxPoolGradWithArgmaxInputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
  }
  size_t output_num = outputs.size();
  if (output_num != kMaxPoolGradWithArgmaxOutputsNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }
  x_type_size_ = GetTypeByte(TypeIdToType(inputs[kIndex0]->GetDtype()));
  dy_type_size_ = GetTypeByte(TypeIdToType(inputs[kIndex1]->GetDtype()));
  idx_type_size_ = GetTypeByte(TypeIdToType(inputs[kIndex2]->GetDtype()));
  dx_type_size_ = GetTypeByte(TypeIdToType(outputs[kIndex0]->GetDtype()));
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

using KernelRunFunc = MaxPoolWithArgmaxGradGpuKernelMod::KernelRunFunc;
// int the python api description, input data type is number but CalExtractImagePatchesNHWC only support four type.
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MaxPoolWithArgmaxGradGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolWithArgmaxGradGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolWithArgmaxGradGpuKernelMod::LaunchKernel<half, int>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolGradWithArgmax, MaxPoolWithArgmaxGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
