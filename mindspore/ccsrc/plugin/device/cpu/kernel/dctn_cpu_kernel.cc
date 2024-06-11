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

#include "plugin/device/cpu/kernel/dctn_cpu_kernel.h"
#include "ops/op_utils.h"
#include "kernel/kernel.h"
#include "mindspore/core/mindapi/base/types.h"
#include "utils/fft_helper.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr double kDCTFactor = 2.0;
constexpr int64_t kNormFactor = 2;
constexpr int64_t kDCTType = 2;
constexpr int64_t kIDCTType = 3;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool DCTNCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void DCTNCpuKernelMod::ResetResource() {
  dim_.clear();
  s_.clear();
  tensor_shape_.clear();
  calculate_shape_.clear();
}

void DCTNCpuKernelMod::DCTNGetAttr() {
  if (s_is_none_ && !dim_is_none_) {
    for (size_t i = 0; i < dim_.size(); i++) {
      (void)s_.emplace_back(tensor_shape_[dim_[i]]);
    }
  }
  if (dim_is_none_ && !s_is_none_) {
    for (size_t i = 0; i < s_.size(); i++) {
      (void)dim_.emplace_back(x_rank_ - s_.size() + i);
    }
  }
  if (dim_is_none_ && s_is_none_) {
    for (int64_t i = 0; i < x_rank_; i++) {
      (void)dim_.emplace_back(i);
      (void)s_.emplace_back(tensor_shape_[i]);
    }
  }
}

int DCTNCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();

  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  dct_type_ = (kernel_name_ == prim::kPrimDCTN->name()) ? kDCTType : kIDCTType;

  // Get or set attribute s and dims.
  auto s_opt = inputs[kIndex2]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (s_opt.has_value()) {
    s_ = s_opt.value();
  } else {
    s_is_none_ = true;
  }
  auto dim_opt = inputs[kIndex3]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
    for (size_t i = 0; i < dim_.size(); i++) {
      dim_[i] = dim_[i] < 0 ? x_rank_ + dim_[i] : dim_[i];
    }
  } else {
    dim_is_none_ = true;
  }
  auto norm_opt = inputs[kIndex4]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm_ = static_cast<NormMode>(norm_opt.value());
  } else {
    norm_ = NormMode::ORTHO;
  }
  is_ortho_ = (norm_ == NormMode::ORTHO);

  DCTNGetAttr();

  forward_ = IsForwardOp(kernel_name_);
  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));

  // Calculate the required memory based on s and dim.
  calculate_shape_.clear();
  calculate_element_nums_ = GetCalculateElementNum(tensor_shape_, dim_, s_, input_element_nums_);
  for (size_t i = 0; i < tensor_shape_.size(); i++) {
    (void)calculate_shape_.emplace_back(tensor_shape_[i]);
  }
  for (size_t i = 0; i < dim_.size(); i++) {
    calculate_shape_[dim_[i]] = s_[i];
  }

  fft_nums_ = std::accumulate(s_.begin(), s_.end(), 1, std::multiplies<int64_t>());
  auto cmpt_nums = fft_nums_ * pow(kNormFactor, static_cast<int64_t>(s_.size()));
  norm_weight_ = GetNormalized(cmpt_nums, norm_, forward_);
  return KRET_OK;
}

template <typename T_in, typename T_out>
bool DCTNCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                    const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

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
  PocketFFTDCT<T_out>(calculate_input, output_ptr, dct_type_, fct, calculate_shape_, dim_, is_ortho_);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return true;
}

template <typename T_in, typename T_out>
bool DCTNCpuKernelMod::LaunchKernelComplex(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_in *>(outputs[kIndex0]->device_ptr());

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
  T_out *output_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  T_out *output_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  PocketFFTDCT<T_out>(calculate_input_real, output_real, dct_type_, fct, calculate_shape_, dim_, is_ortho_);
  PocketFFTDCT<T_out>(calculate_input_imag, output_imag, dct_type_, fct, calculate_shape_, dim_, is_ortho_);

  for (int64_t i = 0; i < calculate_element_nums_; ++i) {
    std::complex<T_out> temp_val{*(output_real + i), *(output_imag + i)};
    *(output_ptr + i) = temp_val;
  }

  // Release temporary memory
  free(calculate_input_real);
  free(calculate_input_imag);
  calculate_input_real = nullptr;
  calculate_input_imag = nullptr;
  return true;
}

std::vector<std::pair<KernelAttr, DCTNCpuKernelMod::DCTNFunc>> DCTNCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTNCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTNCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTNCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeBFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTNCpuKernelMod::LaunchKernel<bfloat16, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTNCpuKernelMod::LaunchKernel<Eigen::half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &DCTNCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &DCTNCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &DCTNCpuKernelMod::LaunchKernelComplex<complex64, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeComplex128),
   &DCTNCpuKernelMod::LaunchKernelComplex<complex128, double>}};
std::vector<KernelAttr> DCTNCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DCTNFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DCTN, DCTNCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IDCTN, DCTNCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
