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

#include "plugin/device/cpu/kernel/rfft_grad_cpu_kernel.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "kernel/kernel.h"

#define SWITCH_DIM_CALCULATE(T1, T2)                                                                               \
  if (x_rank_ == 1) {                                                                                              \
    ComputeRFFTGrad<T1, T2, 1>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else if (x_rank_ == 2) {                                                                                       \
    ComputeRFFTGrad<T1, T2, 2>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else if (x_rank_ == 3) {                                                                                       \
    ComputeRFFTGrad<T1, T2, 3>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else if (x_rank_ == 4) {                                                                                       \
    ComputeRFFTGrad<T1, T2, 4>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else if (x_rank_ == 5) {                                                                                       \
    ComputeRFFTGrad<T1, T2, 5>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else if (x_rank_ == 6) {                                                                                       \
    ComputeRFFTGrad<T1, T2, 6>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else if (x_rank_ == 7) {                                                                                       \
    ComputeRFFTGrad<T1, T2, 7>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  } else {                                                                                                         \
    ComputeRFFTGrad<T1, T2, 8>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element_nums, \
                               tensor2_shape_);                                                                    \
  }

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool RFFTGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RFFTGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  tensor2_shape_ = inputs[kIndex1]->GetShapeVector();
  x2_rank_ = SizeToLong(tensor2_shape_.size());

  // Get or set attribute s and dims.
  auto n_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  dim = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  dim = dim < 0 ? x_rank_ + dim : dim;

  n = n_opt.has_value() ? n_opt.value() : tensor2_shape_[dim];

  auto norm_opt = inputs[kIndex4]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm = static_cast<mindspore::NormMode>(norm_opt.value());
  } else {
    norm = NormMode::BACKWARD;
  }

  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));

  return KRET_OK;
}

double RFFTGradGetnormalized(int64_t element_nums, mindspore::NormMode norm_type_) {
  double result = 1.0;
  if (norm_type_ == NormMode::BACKWARD) {
    result = 1.0 * element_nums;
  } else if (norm_type_ == NormMode::ORTHO) {
    result = 1.0 * sqrt(static_cast<double>(element_nums));
  }
  return result;
}

template <typename T_in, typename T_out>
void RFFTGradGenerateCalculateInput(T_in *array_in, T_out *array_out, int64_t element_nums,
                                    const std::vector<int64_t> &x_shape, const std::vector<int64_t> &calculate_shape,
                                    int64_t n, int64_t dim) {
  // compute original and new offsets for each dim
  std::vector<int64_t> offsets(x_shape.size(), 0);
  std::vector<int64_t> new_offsets(x_shape.size(), 0);
  for (size_t j = 0; j < x_shape.size(); j++) {
    offsets[j] = std::accumulate(x_shape.begin() + j + 1, x_shape.end(), 1, std::multiplies<>());
    new_offsets[j] = std::accumulate(calculate_shape.begin() + j + 1, calculate_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < element_nums; ++i) {
    std::vector<int64_t> index(x_shape.size(), 0);
    int64_t flat_index = i;
    // compute original coordinates
    for (size_t dim_index = 0; dim_index < offsets.size(); ++dim_index) {
      index[dim_index] = flat_index / offsets[dim_index];
      flat_index %= offsets[dim_index];
    }
    // if n > input.shape[dim] ->truncate, invalid ele should be dropped out
    if (index[dim] >= n) {
      continue;
    }
    int64_t new_flat_index = 0;
    for (size_t dim_index = 0; dim_index < new_offsets.size(); ++dim_index) {
      new_flat_index += index[dim_index] * new_offsets[dim_index];
    }
    array_out[new_flat_index] = static_cast<T_out>(array_in[i]);
  }
}

template <typename T_in, typename T_out, int x_rank>
bool ComputeRFFTGrad(T_in *input_ptr, T_out *output_ptr, mindspore::NormMode norm_type, int64_t n, int64_t dim,
                     std::vector<int64_t> x_shape, int64_t element_nums, std::vector<int64_t> out_shape) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = x_shape[i];
  }
  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0], tensor_shape);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  Eigen::array<int, 1> dims_array;
  dims_array[0] = dim;
  out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(dims_array);

  // padding or trimmed back to input2's shape
  Eigen::array<Eigen::DenseIndex, x_rank> final_out_shape;
  for (int i = 0; i < x_rank; ++i) {
    final_out_shape[i] = out_shape[i];
  }
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> final_out(final_out_shape);
  final_out.setZero();
  Eigen::DSizes<Eigen::DenseIndex, x_rank> offsets;
  Eigen::array<Eigen::DenseIndex, x_rank> slice_sizes(out.dimensions());
  for (auto i = 0; i < x_rank; i++) {
    offsets[i] = 0;
    slice_sizes[i] = std::min(out_shape[i], static_cast<int64_t>(slice_sizes[i]));
  }
  final_out.slice(offsets, slice_sizes) = out.slice(offsets, slice_sizes);

  double norm_weight = RFFTGradGetnormalized(n, norm_type);

  T_out *out_ptr = final_out.data();
  for (int i = 0; i < final_out.size(); i++) {
    T_out temp_value = *(out_ptr + i);
    temp_value *= norm_weight;
    *(output_ptr + i) = temp_value;
  }

  return true;
}

template <typename T_in, typename T_mid, typename T_out>
bool RFFTGradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

  // Calculate the required memory based on n and dim.
  int64_t calculate_element_nums = input_element_nums_ / tensor_shape_[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape_.begin(), tensor_shape_.end());
  calculate_shape[dim] = n;

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element_nums));
  if (memset_s(calculate_input, sizeof(T_mid) * calculate_element_nums, 0, sizeof(T_mid) * calculate_element_nums) !=
      EOK) {
    free(calculate_input);
    calculate_input = nullptr;
    MS_LOG(EXCEPTION) << kernel_name_ << " memset_s failed. ";
  }
  RFFTGradGenerateCalculateInput<T_in, T_mid>(input_ptr, calculate_input, input_element_nums_, tensor_shape_,
                                              calculate_shape, n, dim);

  // Run FFT according to parameters
  SWITCH_DIM_CALCULATE(T_mid, T_out);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return true;
}

#define ONE_DIM_CPU_REG(MS_Tin1, MS_Tin2, MS_Tout, T_in, T_mid, T_out) \
  KernelAttr()                                                         \
    .AddInputAttr(MS_Tin1)                  /* input1 */               \
    .AddInputAttr(MS_Tin2)                  /* input2 */               \
    .AddOptionalInputAttr(kNumberTypeInt64) /* n */                    \
    .AddInputAttr(kNumberTypeInt64)         /* dim */                  \
    .AddOptionalInputAttr(kNumberTypeInt64) /* norm */                 \
    .AddOutputAttr(MS_Tout),                                           \
    &RFFTGradCpuKernelMod::LaunchKernel<T_in, T_mid, T_out>

std::vector<std::pair<KernelAttr, RFFTGradCpuKernelMod::RFFTGradFunc>> RFFTGradCpuKernelMod::func_list_ = {
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeInt16, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeInt32, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeInt64, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat16, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat32, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat64, kNumberTypeComplex128, complex128, complex128,
                   complex128)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex64, kNumberTypeComplex64, kNumberTypeComplex64, complex64, complex64, complex64)},
  {ONE_DIM_CPU_REG(kNumberTypeComplex128, kNumberTypeComplex128, kNumberTypeComplex128, complex128, complex128,
                   complex128)}};

std::vector<KernelAttr> RFFTGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RFFTGradFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, RFFTGrad, RFFTGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
