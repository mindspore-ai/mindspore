/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/cuda/arrays/reverse_v2_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t index2 = 2;
}
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool ReverseV2GpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
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

int ReverseV2GpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  input_shape_ = inputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input");

  input_rank_ = input_shape_.size();
  if (input_rank_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 1, but got "
                      << input_rank_;
  }
  input_size_ = 1;
  for (size_t i = 0; i < input_rank_; i++) {
    input_size_ *= static_cast<size_t>(input_shape_[i]);
  }

  strides_.resize(input_rank_);
  strides_[input_rank_ - 1] = 1;
  for (int32_t i = input_rank_ - 2; i >= 0; i--) {
    strides_[i] = static_cast<int64_t>(input_shape_[i + 1]) * strides_[i + 1];
  }

  axis_ = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (axis_.size() < 1) {
    return KRET_OK;
  }

  std::transform(axis_.begin(), axis_.end(), axis_.begin(),
                 [&](int64_t dimension) { return dimension < 0 ? dimension + input_rank_ : dimension; });

  workspace_size_list_.push_back(input_rank_ * sizeof(size_t));
  workspace_size_list_.push_back(input_rank_ * sizeof(int64_t));
  workspace_size_list_.push_back(axis_.size() * sizeof(int64_t));

  return KRET_OK;
}

template <typename T>
bool ReverseV2GpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input_device = GetDeviceAddress<T>(inputs, 0);
  T *output_device = GetDeviceAddress<T>(outputs, 0);
  if (axis_.size() < 1) {
    MS_LOG(WARNING)
      << "The 'axis' has no value in it, no need to reverse any dimension on the input. The output is the "
         "same as the input.";
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_device, input_device, input_size_ * sizeof(T), cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync failed in ReverseV2GpuKernelMod::Launch.")
    return true;
  }
  size_t *input_shape_device = GetDeviceAddress<size_t>(workspace, 0);
  int64_t *strides_device = GetDeviceAddress<int64_t>(workspace, 1);
  int64_t *axis_device = GetDeviceAddress<int64_t>(workspace, 2);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape_device, &input_shape_[0], workspace_size_list_[0], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync for input_shape_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(strides_device, &strides_[0], workspace_size_list_[1], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync for strides_ failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(axis_device, &axis_[0], workspace_size_list_[index2], cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync for axis_ failed");

  auto status = CalReverseV2(input_device, output_device, input_shape_device, strides_device, axis_device, input_size_,
                             axis_.size(), reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ReverseV2GpuKernelMod::ReverseV2LaunchFunc>> ReverseV2GpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeBool),
   &ReverseV2GpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &ReverseV2GpuKernelMod::LaunchKernel<Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &ReverseV2GpuKernelMod::LaunchKernel<Complex<double>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &ReverseV2GpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &ReverseV2GpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &ReverseV2GpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt8),
   &ReverseV2GpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt16),
   &ReverseV2GpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt32),
   &ReverseV2GpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeUInt64),
   &ReverseV2GpuKernelMod::LaunchKernel<uint64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt8),
   &ReverseV2GpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt16),
   &ReverseV2GpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt32),
   &ReverseV2GpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &ReverseV2GpuKernelMod::LaunchKernel<int64_t>},
};

std::vector<KernelAttr> ReverseV2GpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, ReverseV2GpuKernelMod::ReverseV2LaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ReverseV2, ReverseV2GpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
