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
#include "kernel/cpu/native/dct_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "ops_utils/op_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "utils/fft_helper.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace kernel {
namespace dct_cpu {
namespace {
constexpr int64_t kNormFactor = 2;
constexpr int64_t kDCTType = 2;
constexpr int64_t kIDCTType = 3;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool DCTCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int DCTCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  // Get or set attribute s and dims.
  dct_type_ = (kernel_name_ == prim::kPrimDCT->name()) ? kDCTType : kIDCTType;

  dim_ = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  dim_ = dim_ < 0 ? x_rank_ + dim_ : dim_;

  auto n_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  if (n_opt.has_value()) {
    n_ = n_opt.value();
  } else {
    n_ = tensor_shape_[dim_];
  }

  auto norm_opt = inputs[kIndex4]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm_ = static_cast<mindspore::NormMode>(norm_opt.value());
  } else {
    norm_ = NormMode::ORTHO;
  }

  forward_ = IsForwardOp(kernel_name_);
  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));
  is_ortho_ = (norm_ == NormMode::ORTHO);
  UpdateParam();
  return KRET_OK;
}

void DCTCpuKernelMod::UpdateParam() {
  calculate_shape_.clear();
  calculate_element_nums_ = input_element_nums_ / tensor_shape_[dim_] * n_;
  for (size_t i = 0; i < tensor_shape_.size(); i++) {
    (void)calculate_shape_.emplace_back(tensor_shape_[i]);
  }
  calculate_shape_[dim_] = n_;
  auto cmpt_num = kNormFactor * n_;
  norm_weight_ = GetNormalized(cmpt_num, norm_, forward_);
}

template <typename T_in, typename T_out>
bool DCTCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = GetDeviceAddress<T_in>(inputs, kIndex0);
  auto *output_ptr = GetDeviceAddress<T_out>(outputs, kIndex0);

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  auto ret =
    memset_s(calculate_input, sizeof(T_out) * calculate_element_nums_, 0, sizeof(T_out) * calculate_element_nums_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }

  ShapeCopy<T_in, T_out>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // Run FFT according to parameters
  std::vector<int64_t> dim(1, dim_);
  PocketFFTDCT<T_out>(calculate_input, output_ptr, dct_type_, fct, calculate_shape_, dim, is_ortho_);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return true;
}

template <typename T_in, typename T_out>
bool DCTCpuKernelMod::LaunchKernelComplex(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = GetDeviceAddress<T_in>(inputs, kIndex0);
  auto *output_ptr = GetDeviceAddress<T_in>(outputs, kIndex0);

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  T_out *calculate_input_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  auto ret_real =
    memset_s(calculate_input_real, sizeof(T_out) * calculate_element_nums_, 0, sizeof(T_out) * calculate_element_nums_);
  auto ret_imag =
    memset_s(calculate_input_imag, sizeof(T_out) * calculate_element_nums_, 0, sizeof(T_out) * calculate_element_nums_);
  if (ret_real != EOK || ret_imag != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret_real, ret_imag =" << ret_real << ","
                      << ret_imag;
  }

  ShapeCopy<T_in, T_out>(input_ptr, calculate_input_real, tensor_shape_, calculate_shape_);
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input_imag, tensor_shape_, calculate_shape_, false);

  // Run FFT according to parameters
  std::vector<int64_t> dim(1, dim_);
  T_out *output_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  T_out *output_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  PocketFFTDCT<T_out>(calculate_input_real, output_real, dct_type_, fct, calculate_shape_, dim, is_ortho_);
  PocketFFTDCT<T_out>(calculate_input_imag, output_imag, dct_type_, fct, calculate_shape_, dim, is_ortho_);

  for (int64_t i = 0; i < calculate_element_nums_; ++i) {
    std::complex<T_out> temp_val{*(output_real + i), *(output_imag + i)};
    *(output_ptr + i) = temp_val;
  }

  // Release temporary memory
  free(calculate_input_real);
  free(calculate_input_imag);
  calculate_input_real = nullptr;
  calculate_input_imag = nullptr;
  free(output_real);
  free(output_imag);
  output_real = nullptr;
  output_imag = nullptr;
  return true;
}

std::vector<std::pair<KernelAttr, DCTCpuKernelMod::DCTFunc>> DCTCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTCpuKernelMod::LaunchKernel<bfloat16, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTCpuKernelMod::LaunchKernel<Eigen::half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &DCTCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &DCTCpuKernelMod::LaunchKernelComplex<complex64, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &DCTCpuKernelMod::LaunchKernelComplex<complex128, double>}};

std::vector<KernelAttr> DCTCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DCTFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DCT, DCTCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IDCT, DCTCpuKernelMod);
}  // namespace dct_cpu
}  // namespace kernel
}  // namespace mindspore
