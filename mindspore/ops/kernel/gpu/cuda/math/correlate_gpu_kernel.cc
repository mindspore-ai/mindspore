/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/gpu/cuda/math/correlate_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool CorrelateGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto prim = primitive_;
  MS_EXCEPTION_IF_NULL(prim);

  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [unit8, uint16, uint32, uint64, int8, "
                  << "int16, int32, int64, float16, float32, float64, complex64, complex128, bool], but got: "
                  << kernel_attr << ".";
    return false;
  }
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  kernel_func_ = func_list_[index].second;
  return true;
}
int CorrelateGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  data_unit_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).dtype);
  std::vector<int64_t> a_shape = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                                      inputs.at(kIndex0)->GetDeviceShapeVector().end());
  size_t a_dims = a_shape.size();
  std::vector<int64_t> v_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                                      inputs.at(kIndex1)->GetDeviceShapeVector().end());
  size_t v_dims = v_shape.size();
  if (a_dims != kIndex1 || v_dims != kIndex1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'a' and 'v' should be 1-D, but got 'a' at"
                      << a_dims << "-D and 'v' at " << v_dims << "-D.";
  }

  a_size_ = a_shape[0];
  v_size_ = v_shape[0];
  if (a_size_ == 0 || v_size_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', input 'a' and 'v' should not be empty, but got 'a' at ("
                      << a_size_ << ") and 'v' at (" << v_dims << ").";
  }
  a_ge_v_ = a_size_ >= v_size_;
  if (a_ge_v_) {
    long_size_ = a_size_;
    short_size_ = v_size_;
  } else {
    long_size_ = v_size_;
    short_size_ = a_size_;
  }
  mode_type_ = static_cast<mindspore::PadMode>(inputs[kIndex2]->GetValueWithCheck<int64_t>());

  out_size_ = long_size_ - short_size_ + 1;
  padded_long_size_ = long_size_;
  copy_start_idx_ = 0;
  if (mode_type_ == mindspore::PadMode::SAME) {
    padded_long_size_ = long_size_ + short_size_ - 1;
    out_size_ = long_size_;
    copy_start_idx_ = short_size_ / 2;
  } else if (mode_type_ == mindspore::PadMode::FULL) {
    padded_long_size_ = long_size_ + 2 * (short_size_ - 1);
    out_size_ = long_size_ + short_size_ - 1;
    copy_start_idx_ = short_size_ - 1;
  }

  workspace_size_list_ = {
    a_size_ * data_unit_size_,   v_size_ * data_unit_size_, padded_long_size_ * data_unit_size_,
    out_size_ * data_unit_size_, v_size_ * data_unit_size_, sizeof(size_t),
  };

  output_size_list_ = {out_size_ * data_unit_size_};
  return KRET_OK;
}

template <typename T_in, typename T_out>
bool CorrelateGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  T_in *a_array = GetDeviceAddress<T_in>(inputs, kIndex0);
  T_in *v_array = GetDeviceAddress<T_in>(inputs, kIndex1);
  T_out *output_array = GetDeviceAddress<T_out>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(a_array);
  MS_EXCEPTION_IF_NULL(v_array);
  MS_EXCEPTION_IF_NULL(output_array);

  T_out *casted_a_array = GetDeviceAddress<T_out>(workspace, kIndex0);
  T_out *casted_v_array = GetDeviceAddress<T_out>(workspace, kIndex1);
  T_out *padded_long_array = GetDeviceAddress<T_out>(workspace, kIndex2);
  T_out *reverse_out_array = GetDeviceAddress<T_out>(workspace, kIndex3);
  MS_EXCEPTION_IF_NULL(casted_a_array);
  MS_EXCEPTION_IF_NULL(casted_v_array);
  MS_EXCEPTION_IF_NULL(padded_long_array);
  MS_EXCEPTION_IF_NULL(reverse_out_array);

  auto status_cast_a = Cast(a_size_, a_array, casted_a_array, stream);
  CHECK_CUDA_STATUS(status_cast_a, kernel_name_);
  auto status_cast_v = Cast(v_size_, v_array, casted_v_array, stream);
  CHECK_CUDA_STATUS(status_cast_v, kernel_name_);

  T_out *long_array = casted_a_array;
  T_out *short_array = casted_v_array;
  if (!a_ge_v_) {
    long_array = casted_v_array;
    short_array = casted_a_array;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(padded_long_array, 0, padded_long_size_, stream),
                                     "Init padded long array with cudamemset failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(padded_long_array + copy_start_idx_, long_array,
                                                     sizeof(T_out) * long_size_, cudaMemcpyDeviceToDevice, stream),
                                     "Copy long array to padded long array failed");

  auto status_conv = CorrelateCalc(padded_long_array, short_array, reverse_out_array, padded_long_size_, short_size_,
                                   mode_type_, device_id_, stream);
  CHECK_CUDA_STATUS(status_conv, kernel_name_);

  if (a_ge_v_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_array, reverse_out_array, sizeof(T_out) * out_size_, cudaMemcpyDeviceToDevice, stream),
      "Copy long array to padded long array failed");
  } else {
    size_t *out_size_device = GetDeviceAddress<size_t>(workspace, kIndex4);
    MS_EXCEPTION_IF_NULL(out_size_device);

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(out_size_device, &out_size_, workspace_size_list_[kIndex4], cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync for output_shape_ failed");

    auto status = CalReverse1D(reverse_out_array, output_array, out_size_device, out_size_, device_id_, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}

template <typename T>
bool CorrelateGpuKernelMod::LaunchKernelComplex(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  T *a_array = GetDeviceAddress<T>(inputs, kIndex0);
  T *v_array = GetDeviceAddress<T>(inputs, kIndex1);
  T *output_array = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(a_array);
  MS_EXCEPTION_IF_NULL(v_array);
  MS_EXCEPTION_IF_NULL(output_array);

  T *v_conj_array = GetDeviceAddress<T>(workspace, kIndex1);
  T *padded_long_array = GetDeviceAddress<T>(workspace, kIndex2);
  T *reverse_out_array = GetDeviceAddress<T>(workspace, kIndex3);
  MS_EXCEPTION_IF_NULL(v_conj_array);
  MS_EXCEPTION_IF_NULL(padded_long_array);
  MS_EXCEPTION_IF_NULL(reverse_out_array);

  auto status_conj = CalConj(v_array, v_conj_array, v_size_, device_id_, stream);
  CHECK_CUDA_STATUS(status_conj, kernel_name_);

  T *long_array = a_array;
  T *short_array = v_conj_array;
  if (!a_ge_v_) {
    long_array = v_conj_array;
    short_array = a_array;
  }

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemsetAsync(padded_long_array, 0, padded_long_size_, stream),
                                     "Init padded long array with cudamemset failed");

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(padded_long_array + copy_start_idx_, long_array,
                                                     sizeof(T) * long_size_, cudaMemcpyDeviceToDevice, stream),
                                     "Copy long array to padded long array failed");

  auto status_conv = CorrelateCalc(padded_long_array, short_array, reverse_out_array, padded_long_size_, short_size_,
                                   mode_type_, device_id_, stream);
  CHECK_CUDA_STATUS(status_conv, kernel_name_);

  if (a_ge_v_) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(output_array, reverse_out_array, sizeof(T) * out_size_, cudaMemcpyDeviceToDevice, stream),
      "Copy long array to padded long array failed");
  } else {
    size_t *out_size_device = GetDeviceAddress<size_t>(workspace, kIndex4);
    MS_EXCEPTION_IF_NULL(out_size_device);

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(out_size_device, &out_size_, workspace_size_list_[kIndex4], cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync for output_shape_ failed");

    auto status = CalReverse1D(reverse_out_array, output_array, out_size_device, out_size_, device_id_, stream);
    CHECK_CUDA_STATUS(status, kernel_name_);
  }
  return true;
}
template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, CorrelateGpuKernelMod::CorrelateFunc>> CorrelateGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &CorrelateGpuKernelMod::LaunchKernel<int8_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &CorrelateGpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &CorrelateGpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &CorrelateGpuKernelMod::LaunchKernel<int64_t, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16),
   &CorrelateGpuKernelMod::LaunchKernel<half, half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &CorrelateGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &CorrelateGpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &CorrelateGpuKernelMod::LaunchKernelComplex<Complex<float>>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &CorrelateGpuKernelMod::LaunchKernelComplex<Complex<double>>},
};

std::vector<KernelAttr> CorrelateGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CorrelateFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Correlate, CorrelateGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
