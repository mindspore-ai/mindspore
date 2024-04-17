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

#define FFTN_INPUT_DIM_CASE(T1, T2)                                                                        \
  if (x_rank_ == 1) {                                                                                      \
    EigenFFTNBase<T1, T2, 1>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else if (x_rank_ == 2) {                                                                               \
    EigenFFTNBase<T1, T2, 2>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else if (x_rank_ == 3) {                                                                               \
    EigenFFTNBase<T1, T2, 3>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else if (x_rank_ == 4) {                                                                               \
    EigenFFTNBase<T1, T2, 4>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else if (x_rank_ == 5) {                                                                               \
    EigenFFTNBase<T1, T2, 5>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else if (x_rank_ == 6) {                                                                               \
    EigenFFTNBase<T1, T2, 6>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else if (x_rank_ == 7) {                                                                               \
    EigenFFTNBase<T1, T2, 7>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  } else {                                                                                                 \
    EigenFFTNBase<T1, T2, 8>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape_, dim_); \
  }

namespace mindspore {
namespace kernel {
namespace {
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
  forward_ = IsForwardOp(kernel_name_);
  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));

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

  // Calculate the required memory based on s and dim.
  calculate_element_nums_ = GetCalculateElementNum(tensor_shape_, dim_, s_, input_element_nums_);
  for (size_t i = 0; i < tensor_shape_.size(); i++) {
    (void)calculate_shape_.emplace_back(tensor_shape_[i]);
  }
  for (size_t i = 0; i < dim_.size(); i++) {
    calculate_shape_[dim_[i]] = s_[i];
  }
  int64_t fft_nums = std::accumulate(s_.begin(), s_.end(), 1, std::multiplies<int64_t>());
  norm_weight_ = GetNormalized(fft_nums, norm_, forward_);
  return KRET_OK;
}

template <typename T_in, typename T_out, int x_rank>
bool EigenFFTNBase(T_in *input_ptr, T_out *output_ptr, bool forward, double norm_weight,
                   std::vector<int64_t> calculate_shape, std::vector<int64_t> dim) {
  Eigen::array<Eigen::DenseIndex, x_rank> calculate_shape_array;
  for (size_t i = 0; i < x_rank; ++i) {
    calculate_shape_array[i] = calculate_shape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0],
                                                                                     calculate_shape_array);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  std::vector<int32_t> eigem_dim;
  for (size_t i = 0; i < dim.size(); i++) {
    (void)eigem_dim.emplace_back(static_cast<int32_t>(dim[i]));
  }

  if (forward) {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(eigem_dim);
  } else {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(eigem_dim);
  }

  T_out *out_ptr = out.data();
  for (int i = 0; i < out.size(); i++) {
    T_out temp_value = *(out_ptr + i);
    temp_value *= norm_weight;
    *(output_ptr + i) = temp_value;
  }
  return true;
}

template <typename T_in, typename T_mid, typename T_out>
bool FFTNBaseCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element_nums_));
  auto ret =
    memset_s(calculate_input, sizeof(T_mid) * calculate_element_nums_, 0, sizeof(T_mid) * calculate_element_nums_);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  ShapeCopy<T_in, T_mid>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // Run FFT according to parameters
  FFTN_INPUT_DIM_CASE(T_mid, T_out);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return true;
}

#define ONE_DIM_CPU_REG(MS_Tin, MS_Tout, T_in, T_mid, T_out) \
  KernelAttr()                                               \
    .AddInputAttr(MS_Tin)                   /* input */      \
    .AddOptionalInputAttr(kNumberTypeInt64) /* s */          \
    .AddOptionalInputAttr(kNumberTypeInt64) /* dim */        \
    .AddOptionalInputAttr(kNumberTypeInt64) /* norm */       \
    .AddOutputAttr(MS_Tout),                                 \
    &FFTNBaseCpuKernelMod::LaunchKernel<T_in, T_mid, T_out>

std::vector<std::pair<KernelAttr, FFTNBaseCpuKernelMod::FFTNBaseFunc>> FFTNBaseCpuKernelMod::func_list_ = {
  {ONE_DIM_CPU_REG(kNumberTypeInt16, kNumberTypeComplex64, int16_t, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeInt32, kNumberTypeComplex64, int32_t, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeInt64, kNumberTypeComplex64, int64_t, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat16, kNumberTypeComplex64, Eigen::half, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat32, kNumberTypeComplex64, float, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat64, kNumberTypeComplex128, double, double, complex128)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, complex128, complex128, complex128)}};

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
}  // namespace kernel
}  // namespace mindspore
