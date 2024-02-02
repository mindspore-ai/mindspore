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
#include <algorithm>
#include "ops/op_utils.h"
#include "kernel/kernel.h"

#define SWITCH_DIM_CALCULATE(T1, T2)                                                                \
  if (x_rank_ == 1) {                                                                               \
    ComputeFFTBase<T1, T2, 1>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else if (x_rank_ == 2) {                                                                        \
    ComputeFFTBase<T1, T2, 2>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else if (x_rank_ == 3) {                                                                        \
    ComputeFFTBase<T1, T2, 3>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else if (x_rank_ == 4) {                                                                        \
    ComputeFFTBase<T1, T2, 4>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else if (x_rank_ == 5) {                                                                        \
    ComputeFFTBase<T1, T2, 5>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else if (x_rank_ == 6) {                                                                        \
    ComputeFFTBase<T1, T2, 6>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else if (x_rank_ == 7) {                                                                        \
    ComputeFFTBase<T1, T2, 7>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
  } else {                                                                                          \
    ComputeFFTBase<T1, T2, 8>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape_, \
                              calculate_element_nums_);                                             \
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
  auto n_opt = inputs[kIndex1]->GetOptionalValueWithCheck<int64_t>();
  dim = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  dim = dim < 0 ? x_rank_ + dim : dim;

  n = n_opt.has_value() ? n_opt.value() : tensor_shape_[dim];

  auto norm_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm = static_cast<mindspore::NormMode>(norm_opt.value());
  } else {
    norm = NormMode::BACKWARD;
  }

  if (kernel_name_ == prim::kPrimIFFT->name()) {
    forward = false;
  } else {
    forward = true;
  }

  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));

  return KRET_OK;
}

double Getnormalized(int64_t element_nums_, mindspore::NormMode norm_type_, bool forward) {
  double result = 1.0;
  if (forward) {
    if (norm_type_ == NormMode::FORWARD) {
      result = 1.0 / element_nums_;
    } else if (norm_type_ == NormMode::ORTHO) {
      result = 1.0 / sqrt(static_cast<double>(element_nums_));
    }
  } else {
    if (norm_type_ == NormMode::FORWARD) {
      result = 1.0 * element_nums_;
    } else if (norm_type_ == NormMode::ORTHO) {
      result = 1.0 * sqrt(static_cast<double>(element_nums_));
    }
  }
  return result;
}

template <typename T_in, typename T_out>
void GenarateCalculateInput(T_in *array_in, T_out *array_out, int64_t element_nums_,
                            const std::vector<int64_t> &x_shape, const std::vector<int64_t> &calculate_shape, int64_t n,
                            int64_t dim) {
  // compute original and new offsets for each dim
  std::vector<int64_t> offsets(x_shape.size(), 0);
  std::vector<int64_t> new_offsets(x_shape.size(), 0);
  for (size_t j = 0; j < x_shape.size(); j++) {
    offsets[j] = std::accumulate(x_shape.begin() + j + 1, x_shape.end(), 1, std::multiplies<>());
    new_offsets[j] = std::accumulate(calculate_shape.begin() + j + 1, calculate_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < element_nums_; ++i) {
    std::vector<int64_t> index(x_shape.size(), 0);
    int64_t flat_index = i;
    // compute original coordinates
    for (size_t dim = 0; dim < offsets.size(); ++dim) {
      index[dim] = flat_index / offsets[dim];
      flat_index %= offsets[dim];
    }
    // if n > input.shape[dim] ->truncate, invalid ele should be dropped out
    if (index[dim] >= n) {
      continue;
    }
    int64_t new_flat_index = 0;
    for (size_t dim = 0; dim < new_offsets.size(); ++dim) {
      new_flat_index += index[dim] * new_offsets[dim];
    }
    array_out[new_flat_index] = static_cast<T_out>(array_in[i]);
  }
}

template <typename T_in, typename T_out, int x_rank>
bool ComputeFFTBase(T_in *input_ptr, T_out *output_ptr, bool forward, mindspore::NormMode norm_type, int64_t n,
                    int64_t dim, std::vector<int64_t> tensor_shape_, int64_t element_nums_) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = tensor_shape_[i];
  }
  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0], tensor_shape);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  Eigen::array<int, 1> dims_array;
  dims_array[0] = dim;

  if (forward) {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(dims_array);
  } else {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(dims_array);
  }

  double norm_weight = Getnormalized(tensor_shape_[dim], norm_type, forward);

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
  int64_t calculate_element_nums_ = input_element_nums_ / tensor_shape_[dim] * n;
  std::vector<int64_t> calculate_shape_(tensor_shape_.begin(), tensor_shape_.end());
  calculate_shape_[dim] = n;

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element_nums_));
  // memset(calculate_input, 0, sizeof(T_mid) * calculate_element_nums_);
  memset_s(calculate_input, sizeof(T_mid) * calculate_element_nums_, 0, sizeof(T_mid) * calculate_element_nums_);
  GenarateCalculateInput<T_in, T_mid>(input_ptr, calculate_input, input_element_nums_, tensor_shape_, calculate_shape_,
                                      n, dim);

  // Run FFT according to parameters
  SWITCH_DIM_CALCULATE(T_mid, T_out);

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
