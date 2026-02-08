/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/fftbase_cpu_kernel.h"
#include "mindapi/base/type_id.h"
#include "ops_utils/op_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "utils/fft_helper.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace kernel {
namespace fftbase_cpu {
namespace {
const int64_t kNum2 = 2;
constexpr int kOnesideDivisor = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool FFTBaseCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void FFTBaseCpuKernelMod::ResetResource() {
  tensor_shape_.clear();
  calculate_shape_.clear();
}

void FFTBaseCpuKernelMod::UpdateParam() {
  calculate_shape_.clear();
  calculate_element_nums_ = input_element_nums_ / tensor_shape_[dim_] * n_;
  for (size_t i = 0; i < tensor_shape_.size(); i++) {
    (void)calculate_shape_.emplace_back(tensor_shape_[i]);
  }
  calculate_shape_[dim_] = n_;
  norm_weight_ = GetNormalized(n_, norm_, forward_);
}

void FFTBaseCpuKernelMod::ApplyWorkSpace(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  auto out_type_id = outputs[kIndex0]->dtype_id();
  auto workspace_size = calculate_element_nums_ * static_cast<int64_t>(abstract::TypeIdSize(out_type_id));
  if (out_type_id != kNumberTypeComplex64 && out_type_id != kNumberTypeComplex128) {
    workspace_size *= kNum2;
  }
  workspace_size_list_.push_back(workspace_size);
}

int FFTBaseCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  // Get or set attribute s and dims.
  dim_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  dim_ = dim_ < 0 ? x_rank_ + dim_ : dim_;

  auto n_opt = inputs[kIndex1]->GetOptionalValueWithCheck<int64_t>();
  if (n_opt.has_value()) {
    n_ = n_opt.value();
  } else if (kernel_name_ == prim::kPrimHFFT->name()) {
    n_ = (tensor_shape_[dim_] - 1) * kNum2;
  } else {
    n_ = tensor_shape_[dim_];
    if (kernel_name_ == prim::kPrimIRFFT->name() || kernel_name_ == prim::kPrimHFFT->name()) {
      n_ = kOnesideDivisor * (tensor_shape_[dim_] - 1);
    }
  }

  auto norm_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm_ = static_cast<mindspore::NormMode>(norm_opt.value());
  } else {
    norm_ = NormMode::BACKWARD;
  }

  forward_ = IsForwardOp(kernel_name_);
  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));

  UpdateParam();
  ApplyWorkSpace(inputs, outputs);
  return KRET_OK;
}

template <typename T_in, typename T_out>
bool FFTBaseCpuKernelMod::LaunchKernelC2C(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<std::complex<T_out> *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  auto calculate_input = GetDeviceAddress<std::complex<T_out>>(workspace, kIndex0);
  auto ret = memset_s(calculate_input, workspace[kIndex0]->size(), 0, workspace[kIndex0]->size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }

  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // Run FFT according to parameters
  std::vector<int64_t> dim(1, dim_);
  PocketFFTC2C<T_out>(calculate_input, output_ptr, forward_, fct, calculate_shape_, dim);

  return true;
}

template <typename T_in, typename T_out>
bool FFTBaseCpuKernelMod::LaunchKernelR2C(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<std::complex<T_out> *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  auto calculate_input = GetDeviceAddress<T_out>(workspace, kIndex0);
  auto ret = memset_s(calculate_input, workspace[kIndex0]->size(), 0, workspace[kIndex0]->size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // Run FFT according to parameters
  std::vector<int64_t> dim(1, dim_);
  forward_ = kernel_name_ == prim::kPrimIHFFT->name() ? !forward_ : forward_;
  PocketFFTR2C<T_out>(calculate_input, output_ptr, forward_, fct, calculate_shape_, dim);

  if (kernel_name_ == prim::kPrimIHFFT->name()) {
    std::transform(output_ptr, output_ptr + outputs[kIndex0]->size() / sizeof(std::complex<T_out>), output_ptr,
                   [](std::complex<T_out> x) { return std::conj(x); });
  }

  return true;
}

template <typename T_in, typename T_out>
bool FFTBaseCpuKernelMod::LaunchKernelC2R(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  auto calculate_input = GetDeviceAddress<std::complex<T_out>>(workspace, kIndex0);
  auto ret = memset_s(calculate_input, workspace[kIndex0]->size(), 0, workspace[kIndex0]->size());
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  if (kernel_name_ == prim::kPrimHFFT->name()) {
    std::transform(calculate_input, calculate_input + calculate_element_nums_, calculate_input,
                   [](std::complex<T_out> x) { return std::conj(x); });
    forward_ = !forward_;
  }
  // Run FFT according to parameters
  std::vector<int64_t> dim(1, dim_);
  PocketFFTC2R<T_out>(calculate_input, output_ptr, forward_, fct, calculate_shape_, dim);

  return true;
}

template <typename T_in, typename T_out>
bool FFTBaseCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  if (kernel_name_ == prim::kPrimFFT->name() || kernel_name_ == prim::kPrimIFFT->name()) {
    LaunchKernelC2C<T_in, T_out>(inputs, workspace, outputs);
  }
  if (kernel_name_ == prim::kPrimRFFT->name() || kernel_name_ == prim::kPrimIHFFT->name()) {
    LaunchKernelR2C<T_in, T_out>(inputs, workspace, outputs);
  }
  if (kernel_name_ == prim::kPrimIRFFT->name() || kernel_name_ == prim::kPrimHFFT->name()) {
    LaunchKernelC2R<T_in, T_out>(inputs, workspace, outputs);
  }
  return true;
}

std::vector<std::pair<KernelAttr, FFTBaseCpuKernelMod::FFTBaseFunc>> FFTBaseCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernel<bfloat16, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernel<Eigen::half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &FFTBaseCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTBaseCpuKernelMod::LaunchKernelC2C<complex64, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &FFTBaseCpuKernelMod::LaunchKernelC2C<complex128, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernel<bfloat16, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernel<Eigen::half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &FFTBaseCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTBaseCpuKernelMod::LaunchKernelC2R<complex64, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &FFTBaseCpuKernelMod::LaunchKernelC2R<complex128, double>}};

std::vector<KernelAttr> FFTBaseCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTBaseFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFT, FFTBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IFFT, FFTBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RFFT, FFTBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IRFFT, FFTBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HFFT, FFTBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IHFFT, FFTBaseCpuKernelMod);
}  // namespace fftbase_cpu
}  // namespace kernel
}  // namespace mindspore
