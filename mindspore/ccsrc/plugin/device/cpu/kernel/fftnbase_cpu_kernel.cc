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

#include "plugin/device/cpu/kernel/fftnbase_cpu_kernel.h"
#include "ops/op_utils.h"
#include "kernel/kernel.h"
#include "mindspore/core/mindapi/base/types.h"
#include "utils/fft_helper.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kOnesideDivisor = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool FFTNBaseCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void FFTNBaseCpuKernelMod::ResetResource() {
  dim_.clear();
  s_.clear();
  tensor_shape_.clear();
  calculate_shape_.clear();
}

int FFTNBaseCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  ResetResource();

  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  // Get or set attribute s and dims.
  auto s_opt = inputs[kIndex1]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (s_opt.has_value()) {
    s_ = s_opt.value();
  }
  auto dim_opt = inputs[kIndex2]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
    for (size_t i = 0; i < dim_.size(); i++) {
      dim_[i] = dim_[i] < 0 ? x_rank_ + dim_[i] : dim_[i];
    }
  }
  auto norm_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm_ = static_cast<NormMode>(norm_opt.value());
  } else {
    norm_ = NormMode::BACKWARD;
  }

  if (!s_opt.has_value() && dim_opt.has_value()) {
    for (size_t i = 0; i < dim_.size(); i++) {
      (void)s_.emplace_back(tensor_shape_[dim_[i]]);
    }
  }
  if (!dim_opt.has_value() && s_opt.has_value()) {
    for (size_t i = 0; i < s_.size(); i++) {
      (void)dim_.emplace_back(x_rank_ - s_.size() + i);
    }
  }
  if (!s_opt.has_value() && !dim_opt.has_value()) {
    for (int64_t i = 0; i < x_rank_; i++) {
      (void)dim_.emplace_back(i);
      (void)s_.emplace_back(tensor_shape_[i]);
    }
  }
  // if (!s_opt.has_value() && (kernel_name_ == prim::kPrimHFFT2->name() || kernel_name_ == prim::kPrimHFFTN->name() ||
  //                            kernel_name_ == prim::kPrimIRFFT2->name() || kernel_name_ == prim::kPrimIRFFTN->name()))
  //                            {
  //   s_.back() = (s_.back() - 1) * kOnesideDivisor;
  // }

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
  norm_weight_ = GetNormalized(fft_nums_, norm_, forward_);
  return KRET_OK;
}

template <typename T_in, typename T_out>
bool FFTNBaseCpuKernelMod::LaunchKernelC2C(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<std::complex<T_out> *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  std::complex<T_out> *calculate_input =
    static_cast<std::complex<T_out> *>(malloc(sizeof(std::complex<T_out>) * calculate_element_nums_));
  auto ret = memset_s(calculate_input, sizeof(std::complex<T_out>) * calculate_element_nums_, 0,
                      sizeof(std::complex<T_out>) * calculate_element_nums_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }

  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // Run FFT according to parameters
  PocketFFTC2C<T_out>(calculate_input, output_ptr, forward_, fct, calculate_shape_, dim_);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return true;
}

template <typename T_in, typename T_out>
bool FFTNBaseCpuKernelMod::LaunchKernelR2C(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<std::complex<T_out> *>(outputs[kIndex0]->device_ptr());

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
  PocketFFTR2C<T_out>(calculate_input, output_ptr, forward_, fct, calculate_shape_, dim_);

  // if (kernel_name_ == prim::kPrimIHFFT2->name() || kernel_name_ == prim::kPrimIHFFTN->name()) {
  //   std::transform(output_ptr, output_ptr + calculate_element_nums_, output_ptr,
  //                  [](std::complex<T_out> x) { return std::conj(x); });
  // }
  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return true;
}

template <typename T_in, typename T_out>
bool FFTNBaseCpuKernelMod::LaunchKernelC2R(const std::vector<kernel::KernelTensor *> &inputs,
                                           const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on s and dim.
  T_out fct = static_cast<T_out>(norm_weight_);
  // Allocate temporary memory of the required type and size and copy the input into this space.
  std::complex<T_out> *calculate_input =
    static_cast<std::complex<T_out> *>(malloc(sizeof(std::complex<T_out>) * calculate_element_nums_));
  auto ret = memset_s(calculate_input, sizeof(std::complex<T_out>) * calculate_element_nums_, 0,
                      sizeof(std::complex<T_out>) * calculate_element_nums_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // if (kernel_name_ == prim::kPrimHFFT2->name() || kernel_name_ == prim::kPrimHFFTN->name()) {
  //   std::transform(calculate_input, calculate_input + calculate_element_nums_, calculate_input,
  //                  [](std::complex<T_out> x) { return std::conj(x); });
  //   forward_ = !forward_;
  // }
  // Run FFT according to parameters
  PocketFFTC2R<T_out>(calculate_input, output_ptr, forward_, fct, calculate_shape_, dim_);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return true;
}

template <typename T_in, typename T_out>
bool FFTNBaseCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  // if (kernel_name_ == prim::kPrimHFFT2->name() || kernel_name_ == prim::kPrimHFFTN->name() ||
  //     kernel_name_ == prim::kPrimIRFFT2->name() || kernel_name_ == prim::kPrimIRFFTN->name()) {
  //   LaunchKernelC2R<T_in, T_out>(inputs, outputs);
  // }
  // if (kernel_name_ == prim::kPrimRFFT2->name() || kernel_name_ == prim::kPrimRFFTN->name()) {
  //   LaunchKernelR2C<T_in, T_out>(inputs, outputs);
  // }
  // if (kernel_name_ == prim::kPrimIHFFT2->name() || kernel_name_ == prim::kPrimIHFFTN->name()) {
  //   forward_ = !forward_;
  //   LaunchKernelR2C<T_in, T_out>(inputs, outputs);
  // }
  if (kernel_name_ == prim::kPrimFFT2->name() || kernel_name_ == prim::kPrimIFFT2->name() ||
      kernel_name_ == prim::kPrimFFTN->name() || kernel_name_ == prim::kPrimIFFTN->name()) {
    LaunchKernelC2C<T_in, T_out>(inputs, outputs);
  }
  return true;
}

std::vector<std::pair<KernelAttr, FFTNBaseCpuKernelMod::FFTNBaseFunc>> FFTNBaseCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTNBaseCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTNBaseCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTNBaseCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTNBaseCpuKernelMod::LaunchKernel<Eigen::half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTNBaseCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &FFTNBaseCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex64),
   &FFTNBaseCpuKernelMod::LaunchKernelC2C<complex64, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeComplex128),
   &FFTNBaseCpuKernelMod::LaunchKernelC2C<complex128, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTNBaseCpuKernelMod::LaunchKernel<int16_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTNBaseCpuKernelMod::LaunchKernel<int32_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTNBaseCpuKernelMod::LaunchKernel<int64_t, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTNBaseCpuKernelMod::LaunchKernel<Eigen::half, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTNBaseCpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &FFTNBaseCpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat32),
   &FFTNBaseCpuKernelMod::LaunchKernelC2R<complex64, float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOptionalInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat64),
   &FFTNBaseCpuKernelMod::LaunchKernelC2R<complex128, double>}};

std::vector<KernelAttr> FFTNBaseCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTNBaseFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFT2, FFTNBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFTN, FFTNBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IFFT2, FFTNBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IFFTN, FFTNBaseCpuKernelMod);

// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HFFT2, FFTNBaseCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HFFTN, FFTNBaseCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IHFFT2, FFTNBaseCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IHFFTN, FFTNBaseCpuKernelMod);

// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RFFT2, FFTNBaseCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RFFTN, FFTNBaseCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IRFFT2, FFTNBaseCpuKernelMod);
// MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IRFFTN, FFTNBaseCpuKernelMod);

}  // namespace kernel
}  // namespace mindspore
