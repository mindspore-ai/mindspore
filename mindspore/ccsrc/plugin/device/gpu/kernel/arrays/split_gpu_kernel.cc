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

#include "plugin/device/gpu/kernel/arrays/split_gpu_kernel.h"
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/split_impl.cuh"
#include "ops/split.h"

namespace mindspore {
namespace kernel {
constexpr size_t kSplitInputsNum = 1;
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
bool SplitFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                        const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *input = GetDeviceAddress<T>(inputs, 0);
  T **outputs_device = GetDeviceAddress<T *>(workspace, 0);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs_host_[i] = GetDeviceAddress<T>(outputs, i);
  }
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(outputs_device, outputs_host_.get(), sizeof(T *) * output_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "Split opt cudaMemcpyAsync outputs failed");
  SplitKernel(input_size_, axis_step_, all_size_before_axis_, all_size_axis_, input, outputs_device,
              reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool SplitFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSplitInputsNum, kernel_name_);
  kernel_name_ = base_operator->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Split>(base_operator);
  output_num_ = kernel_ptr->get_output_num();
  outputs_host_ = std::make_unique<void *[]>(output_num_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SplitFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs,
                                 const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSplitInputsNum, kernel_name_);
  int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret != KRET_OK) {
    return ret;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::Split>(base_operator);
  axis_ = kernel_ptr->get_axis();
  auto input_shape = inputs[0]->GetShapeVector();
  int dims = SizeToInt(input_shape.size());
  if (axis_ < 0) {
    axis_ += dims;
  }
  std::string origin_data_format = kOpFormat_DEFAULT;
  auto input_format = GetFormatFromEnumToStr(inputs[0]->GetFormat());
  axis_ = AxisTransform(origin_data_format, input_format, axis_);
  (void)CheckParam(inputs, outputs);

  is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
  if (is_null_input_) {
    return KRET_OK;
  }

  for (size_t i = 0; i < outputs.size(); i++) {
    auto output_shape = outputs[i]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      return KRET_OK;
    }
  }

  input_size_ = 1;
  all_size_before_axis_ = 1;
  all_size_axis_ = 1;
  for (int i = 0; i < SizeToInt(input_shape.size()); i++) {
    input_size_ *= LongToSize(input_shape[i]);
    if (i > axis_) {
      all_size_before_axis_ *= LongToSize(input_shape[i]);
      all_size_axis_ *= input_shape[i];
    }
    if (i == axis_) {
      all_size_before_axis_ *= input_shape[i];
    }
  }

  axis_step_ = LongToInt(input_shape[axis_]) / output_num_;
  workspace_size_list_.clear();
  workspace_size_list_.push_back(sizeof(void *) * output_num_);
  return ret;
}

void SplitFwdGpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs,
                                      const std::vector<KernelTensorPtr> &outputs) const {
  auto input_num = inputs.size();
  auto input_shape = inputs[0]->GetShapeVector();
  int dims = SizeToInt(input_shape.size());
  int output_num = SizeToInt(outputs.size());
  if (output_num <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be greater than 0, but got "
                      << output_num;
  }
  if (input_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
  }
  if (dims == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be 0, but got " << dims;
  }
  if (input_shape[axis_] > 0 && output_num_ > input_shape[axis_]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs cannot be greater than "
                      << input_shape[axis_] << ", but got " << output_num_;
  }
  if (output_num_ != output_num) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be " << output_num_ << ", but got "
                      << output_num;
  }
}

std::vector<std::pair<KernelAttr, SplitFwdGpuKernelMod::SplitFunc>> SplitFwdGpuKernelMod::func_list_ = {
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   &SplitFwdGpuKernelMod::LaunchKernel<half>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &SplitFwdGpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &SplitFwdGpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
   &SplitFwdGpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
   &SplitFwdGpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
   &SplitFwdGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &SplitFwdGpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
   &SplitFwdGpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
   &SplitFwdGpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
   &SplitFwdGpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
   &SplitFwdGpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
   &SplitFwdGpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
   &SplitFwdGpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &SplitFwdGpuKernelMod::LaunchKernel<bool>},
};

std::vector<KernelAttr> SplitFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SplitFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Split, SplitFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
