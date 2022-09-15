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

#include "plugin/device/gpu/kernel/nn/maxpool_with_argmax_gpu_kernel.h"
#include "mindspore/core/ops/max_pool_with_argmax.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMaxPoolWithArgmaxInputsNum = 1;
constexpr size_t kMaxPoolWithArgmaxOutputsNum = 2;
}  // namespace

bool MaxPoolWithArgmaxGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolWithArgmax>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast MaxPoolWithArgmax ops failed!";
  }
  kernel_name_ = kernel_ptr->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolWithArgmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolWithArgmaxOutputsNum, kernel_name_);
  std::vector<int> window;
  std::vector<int64_t> window_me = kernel_ptr->get_kernel_size();
  (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (window.size() < kDim3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 3, but got "
                      << window.size();
  }
  window_height_ = window[kIndex1];
  window_width_ = window[kIndex2];
  std::vector<int> stride;

  std::vector<int64_t> stride_me = kernel_ptr->get_strides();
  (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (stride.size() < kDim3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 3, but got "
                      << stride.size();
  }
  stride_height_ = stride[kIndex1];
  stride_width_ = stride[kIndex2];
  pad_mode_ = kernel_ptr->get_pad_mode();
  pad_top_ = 0;
  pad_left_ = 0;
  device_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  return true;
}

int MaxPoolWithArgmaxGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::MaxPoolWithArgmax>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "cast MaxPoolWithArgmax ops failed!";
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolWithArgmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolWithArgmaxOutputsNum, kernel_name_);
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  if (input_shape.size() < kInputDimLowerLimit || output_shape.size() < kOutputDimLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output cannot be less than 4, but "
                      << "got the dimension of input: " << input_shape.size()
                      << ", the dimension of output: " << output_shape.size();
  }
  n_ = LongToInt(input_shape[kInputIndexForN]);
  c_ = LongToInt(input_shape[kInputIndexForC]);
  input_height_ = LongToInt(input_shape[kInputIndexForH]);
  input_width_ = LongToInt(input_shape[kInputIndexForW]);
  output_height_ = LongToInt(output_shape[kOutputIndexForH]);
  output_width_ = LongToInt(output_shape[kOutputIndexForW]);
  if (pad_mode_ == PadMode::SAME) {
    SetPad();
  }
  return static_cast<int>(KRET_OK);
}

using KernelRunFunc = MaxPoolWithArgmaxGpuKernelMod::KernelRunFunc;
// int the python api description, input data type is number but CalExtractImagePatchesNHWC only support four type.
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &MaxPoolWithArgmaxGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeInt32),
     &MaxPoolWithArgmaxGpuKernelMod::LaunchKernel<half, int>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, MaxPoolWithArgmax, MaxPoolWithArgmaxGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
