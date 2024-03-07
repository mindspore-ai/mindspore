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

#include "plugin/device/cpu/kernel/fftbase_cpu_kernel.h"
#include "ops/op_utils.h"
#include "kernel/kernel.h"
#include "utils/fft_helper.h"

#define EIGEN_FFT_INPUT_RANK_CASE(T1, T2)                                                                \
  if (x_rank_ == 1) {                                                                                    \
    EigenFFTBase<T1, T2, 1>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else if (x_rank_ == 2) {                                                                             \
    EigenFFTBase<T1, T2, 2>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else if (x_rank_ == 3) {                                                                             \
    EigenFFTBase<T1, T2, 3>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else if (x_rank_ == 4) {                                                                             \
    EigenFFTBase<T1, T2, 4>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else if (x_rank_ == 5) {                                                                             \
    EigenFFTBase<T1, T2, 5>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else if (x_rank_ == 6) {                                                                             \
    EigenFFTBase<T1, T2, 6>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else if (x_rank_ == 7) {                                                                             \
    EigenFFTBase<T1, T2, 7>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  } else {                                                                                               \
    EigenFFTBase<T1, T2, 8>(calculate_input, output_ptr, forward_, norm_weight_, calculate_shape, dim_); \
  }

namespace mindspore {
namespace kernel {
namespace {
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
  n_ = n_opt.has_value() ? n_opt.value() : tensor_shape_[dim_];

  auto norm_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm_ = static_cast<mindspore::NormMode>(norm_opt.value());
  } else {
    norm_ = NormMode::BACKWARD;
  }

  forward_ = IsForwardOp(kernel_name_);
  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));
  norm_weight_ = GetNormalized(n_, norm_, forward_);
  return KRET_OK;
}

template <typename T_in, typename T_out, int x_rank>
bool EigenFFTBase(T_in *input_ptr, T_out *output_ptr, bool forward, double norm_weight,
                  std::vector<int64_t> calculate_shape, int64_t dim) {
  Eigen::array<Eigen::DenseIndex, x_rank> calculate_shape_array;
  for (size_t i = 0; i < x_rank; ++i) {
    calculate_shape_array[i] = calculate_shape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0],
                                                                                     calculate_shape_array);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  std::vector<int32_t> eigen_dim;
  (void)eigen_dim.emplace_back(static_cast<int32_t>(dim));

  if (forward) {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(eigen_dim);
  } else {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(eigen_dim);
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
bool FFTBaseCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on s and dim.
  int64_t calculate_element_nums = input_element_nums_ / tensor_shape_[dim_] * n_;
  std::vector<int64_t> calculate_shape(tensor_shape_.begin(), tensor_shape_.end());
  calculate_shape[dim_] = n_;

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element_nums));
  auto ret =
    memset_s(calculate_input, sizeof(T_mid) * calculate_element_nums, 0, sizeof(T_mid) * calculate_element_nums);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset_s failed, ret=" << ret;
  }
  ShapeCopy<T_in, T_mid>(input_ptr, calculate_input, tensor_shape_, calculate_shape);

  // Run FFT according to parameters
  EIGEN_FFT_INPUT_RANK_CASE(T_mid, T_out);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return true;
}

#define ONE_DIM_CPU_REG(MS_Tin, MS_Tout, T_in, T_mid, T_out) \
  KernelAttr()                                               \
    .AddInputAttr(MS_Tin)                   /* x */          \
    .AddOptionalInputAttr(kNumberTypeInt64) /* n */          \
    .AddInputAttr(kNumberTypeInt64)         /* dim */        \
    .AddOptionalInputAttr(kNumberTypeInt64) /* norm */       \
    .AddOutputAttr(MS_Tout),                                 \
    &FFTBaseCpuKernelMod::LaunchKernel<T_in, T_mid, T_out>

std::vector<std::pair<KernelAttr, FFTBaseCpuKernelMod::FFTBaseFunc>> FFTBaseCpuKernelMod::func_list_ = {
  {ONE_DIM_CPU_REG(kNumberTypeInt16, kNumberTypeComplex64, int16_t, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeInt32, kNumberTypeComplex64, int32_t, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeInt64, kNumberTypeComplex64, int64_t, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat16, kNumberTypeComplex64, Eigen::half, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat32, kNumberTypeComplex64, float, float, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeFloat64, kNumberTypeComplex128, double, double, complex128)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, complex128, complex128, complex128)}};

std::vector<KernelAttr> FFTBaseCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FFTBaseFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FFT, FFTBaseCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IFFT, FFTBaseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
