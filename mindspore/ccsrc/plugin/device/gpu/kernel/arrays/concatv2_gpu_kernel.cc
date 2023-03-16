/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/arrays/concatv2_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;

const std::vector<std::pair<KernelAttr, ConcatV2FwdGpuKernelMod::KernelRunFunc>> &ConcatV2FwdGpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, ConcatV2FwdGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<Complex<double>>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<uint>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr().AddAllSameAttr(true).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     &ConcatV2FwdGpuKernelMod::LaunchKernel<bool>}};
  return func_list;
}

bool ConcatV2FwdGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                     const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool ConcatV2FwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  if (input_num_ == 0) {
    return true;
  }
  T *output = GetDeviceAddress<T>(outputs, 0);
  T **inputs_device = GetDeviceAddress<T *>(workspace, 0);
  int *len_axis_device = GetDeviceAddress<int>(workspace, 1);
  for (int i = 0; i < input_num_; i++) {
    auto input_index = not_null_input_index_[i];
    inputs_host_[i] = GetDeviceAddress<T>(inputs, input_index);
  }
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(inputs_device, inputs_host_.data(), sizeof(T *) * input_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "ConcatV2 opt cudaMemcpyAsync inputs failed");
  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemcpyAsync(len_axis_device, len_axis_.data(), sizeof(int) * input_num_, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr_)),
    "ConcatV2 opt cudaMemcpyAsync length on axis failed");
  output_size_ = output_size_list_[0] / sizeof(T);
  ConcatKernel(output_size_, input_num_, all_size_before_axis_, all_size_axis_, len_axis_device, inputs_device, output,
               reinterpret_cast<cudaStream_t>(stream_ptr_));
  return true;
}

bool ConcatV2FwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  ori_axis_ = GetValue<int64_t>(prim->GetAttr("axis"));
  origin_data_format_ = GetValue<std::string>(prim->GetAttr("operator_origin_format"));
  len_axis_.resize(inputs.size());
  return true;
}

int ConcatV2FwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  auto input_0_shape = inputs[0]->GetDeviceShapeAdaptively();
  int dims = SizeToInt(input_0_shape.size());
  axis_ = ori_axis_;
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' must be in the range [-" << dims << "," << dims
                      << "), but got " << axis_;
  }
  if (axis_ < 0) {
    axis_ += dims;
  }
  auto input_format = mindspore::FormatEnumToString(inputs[0]->GetFormat());
  axis_ = AxisTransform(origin_data_format_, input_format, axis_);

  not_null_input_index_.clear();
  len_axis_.clear();
  input_num_ = inputs.size();
  for (int i = 0; i < input_num_; i++) {
    auto input_shape = inputs[i]->GetDeviceShapeAdaptively();
    auto is_null_input = CHECK_NULL_INPUT(input_shape);
    if (!is_null_input) {
      not_null_input_index_.push_back(i);
      len_axis_.push_back(LongToInt(input_shape[axis_]));
    }
  }
  input_num_ = not_null_input_index_.size();
  workspace_size_list_.push_back(sizeof(void *) * input_num_);
  workspace_size_list_.push_back(sizeof(int) * input_num_);
  inputs_host_.resize(input_num_);

  auto output_shape = outputs[0]->GetDeviceShapeAdaptively();
  all_size_before_axis_ = 1;
  all_size_axis_ = 1;
  for (int i = 0; i < SizeToInt(output_shape.size()); i++) {
    if (i > axis_) {
      all_size_before_axis_ *= LongToInt(output_shape[i]);
      all_size_axis_ *= LongToInt(output_shape[i]);
    }
    if (i == axis_) {
      all_size_before_axis_ *= LongToInt(output_shape[i]);
    }
  }
  return KRET_OK;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Concat, ConcatV2FwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
