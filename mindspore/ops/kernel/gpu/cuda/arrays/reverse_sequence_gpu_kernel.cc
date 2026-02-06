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

#include "kernel/gpu/cuda/arrays/reverse_sequence_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "mindspore/ops/infer/reverse_sequence.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool ReverseSequenceGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 2;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  seq_dim_ = GetValue<int64_t>(primitive_->GetAttr("seq_dim"));
  batch_dim_ = GetValue<int64_t>(primitive_->GetAttr("batch_dim"));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int ReverseSequenceGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  auto seq_len_shape = inputs[kIndex1]->GetShapeVector();
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape_, kernel_name_, "x") || CHECK_SHAPE_NULL(seq_len_shape, kernel_name_, "seq_lengths");

  if (input_shape_.size() < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 1, but got "
                      << input_shape_.size();
  }
  shape_size_ = input_shape_.size();  // required for calls
  input_size_ = SizeOf(input_shape_);

  // get seq len shape
  seq_len_size_ = seq_len_shape.size();
  output_size_ = input_size_;  // size does not change

  size_t total_threads = GET_BLOCKS(input_size_) * GET_THREADS;
  total_index_dim_ = total_threads * shape_size_;

  workspace_size_list_.push_back(shape_size_ * sizeof(size_t));       // input_shape
  workspace_size_list_.push_back(shape_size_ * sizeof(size_t));       // cumulative shape
  workspace_size_list_.push_back(total_index_dim_ * sizeof(size_t));  // scratch memory for holding indices per thread
  return KRET_OK;
}

template <typename T, typename S>
bool ReverseSequenceGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *input = GetDeviceAddress<T>(inputs, 0);
  S *seq_len = GetDeviceAddress<S>(inputs, 1);
  constexpr size_t cur_pos_index = 2;
  size_t *input_shape_ptr = GetDeviceAddress<size_t>(workspace, 0);
  size_t *input_cum_shape_ptr = GetDeviceAddress<size_t>(workspace, 1);
  size_t *cur_pos_arr = GetDeviceAddress<size_t>(workspace, cur_pos_index);
  T *output = GetDeviceAddress<T>(outputs, 0);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(input_shape_ptr, &input_shape_[0], input_shape_.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr)),
    "cudaMemcpyAsync input_shape_ failed");
  auto status =
    CalReverseSequence(input_size_, input, seq_len, batch_dim_, seq_dim_, cur_pos_arr, input_shape_ptr,
                       input_cum_shape_ptr, shape_size_, output, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, ReverseSequenceGpuKernelMod::ReverseSequenceLaunchFunc>>
  ReverseSequenceGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
     &ReverseSequenceGpuKernelMod::LaunchKernel<int8_t, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
     &ReverseSequenceGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
     &ReverseSequenceGpuKernelMod::LaunchKernel<int16_t, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
     &ReverseSequenceGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     &ReverseSequenceGpuKernelMod::LaunchKernel<int, int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
     &ReverseSequenceGpuKernelMod::LaunchKernel<int, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &ReverseSequenceGpuKernelMod::LaunchKernel<half, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &ReverseSequenceGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &ReverseSequenceGpuKernelMod::LaunchKernel<float, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &ReverseSequenceGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &ReverseSequenceGpuKernelMod::LaunchKernel<double, int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &ReverseSequenceGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     &ReverseSequenceGpuKernelMod::LaunchKernel<bool, int>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     &ReverseSequenceGpuKernelMod::LaunchKernel<bool, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint8_t, int>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint16_t, int>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint32_t, int>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint64_t, int>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
     &ReverseSequenceGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
     &ReverseSequenceGpuKernelMod::LaunchKernel<Complex<float>, int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeComplex128),
     &ReverseSequenceGpuKernelMod::LaunchKernel<Complex<double>, int>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeComplex64),
     &ReverseSequenceGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &ReverseSequenceGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
};

std::vector<KernelAttr> ReverseSequenceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReverseSequenceGpuKernelMod::ReverseSequenceLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ReverseSequence, ReverseSequenceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
