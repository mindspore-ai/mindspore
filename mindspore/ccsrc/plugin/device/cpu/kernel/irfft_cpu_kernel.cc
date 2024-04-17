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

#include "plugin/device/cpu/kernel/irfft_cpu_kernel.h"
#include <algorithm>
#include "ops/op_utils.h"
#include "kernel/kernel.h"

#define SWITCH_DIM_CALCULATE(T1, T2)                                                   \
  if (x_rank_ == 1) {                                                                  \
    ComputeIRFFT<T1, T2, 1>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else if (x_rank_ == 2) {                                                           \
    ComputeIRFFT<T1, T2, 2>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else if (x_rank_ == 3) {                                                           \
    ComputeIRFFT<T1, T2, 3>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else if (x_rank_ == 4) {                                                           \
    ComputeIRFFT<T1, T2, 4>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else if (x_rank_ == 5) {                                                           \
    ComputeIRFFT<T1, T2, 5>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else if (x_rank_ == 6) {                                                           \
    ComputeIRFFT<T1, T2, 6>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else if (x_rank_ == 7) {                                                           \
    ComputeIRFFT<T1, T2, 7>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  } else {                                                                             \
    ComputeIRFFT<T1, T2, 8>(calculate_input, output_ptr, tensor_shape_, n, dim, norm); \
  }

namespace mindspore {
namespace kernel {
namespace {
constexpr int kRealFFTSideNum = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace

bool IRFFTCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " valid cpu kernel does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int IRFFTCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  tensor_shape_ = inputs[kIndex0]->GetShapeVector();
  x_rank_ = SizeToLong(tensor_shape_.size());

  // Get or set attribute s and dims.
  auto n_opt = inputs[kIndex1]->GetOptionalValueWithCheck<int64_t>();
  dim = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  dim = dim < 0 ? x_rank_ + dim : dim;

  n = n_opt.has_value() ? n_opt.value() : kRealFFTSideNum * (tensor_shape_[dim] - 1);

  auto norm_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  if (norm_opt.has_value()) {
    norm = static_cast<mindspore::NormMode>(norm_opt.value());
  } else {
    norm = NormMode::BACKWARD;
  }

  input_element_nums_ = SizeToLong(SizeOf(tensor_shape_));

  return KRET_OK;
}

double GetNormalizeWeight(int64_t element_nums, mindspore::NormMode norm_type) {
  double result = 1.0;
  if (norm_type == NormMode::FORWARD) {
    result = 1.0 * element_nums;
  } else if (norm_type == NormMode::ORTHO) {
    result = 1.0 * sqrt(static_cast<double>(element_nums));
  }
  return result;
}

template <typename T_in, typename T_out>
void GenerateCalculateInput(T_in *array_in, T_out *array_out, int64_t element_nums) {
  for (int64_t i = 0; i < element_nums; ++i) {
    array_out[i] = static_cast<T_out>(array_in[i]);
  }
}

template <typename T1, typename T2, int x_rank>
Eigen::Tensor<T1, x_rank, Eigen::RowMajor> ReconstructTensor(
  Eigen::array<Eigen::DenseIndex, x_rank> temp_tensor_shape,
  Eigen::TensorMap<Eigen::Tensor<T1, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in, const std::vector<int64_t> &x_shape,
  int n, int dim) {
  // Reconstruct the full fft tensor: temp_tensor
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> temp_tensor(temp_tensor_shape);
  temp_tensor.setZero();

  Eigen::array<Eigen::DenseIndex, x_rank> zero_offsets;
  for (int i = 0; i < x_rank; ++i) {
    zero_offsets[i] = 0;
  }
  Eigen::array<Eigen::DenseIndex, x_rank> input_slice_sizes(in.dimensions());
  // for n that less than input.shape[dim]
  auto oneside_num = n / kRealFFTSideNum + 1;
  input_slice_sizes[dim] = std::min(oneside_num, static_cast<int>(x_shape[dim]));
  temp_tensor.slice(zero_offsets, input_slice_sizes) = in.slice(zero_offsets, input_slice_sizes);

  // rebuild data along the dim with symmetrical data
  if (temp_tensor_shape[dim] - input_slice_sizes[dim] > 0) {
    Eigen::array<bool, x_rank> reverse_dim;
    for (auto i = 0; i < x_rank; i++) {
      reverse_dim[i] = i == dim;
    }
    auto reverse_size = input_slice_sizes;
    reverse_size[dim] = temp_tensor_shape[dim] - input_slice_sizes[dim];
    Eigen::array<Eigen::DenseIndex, x_rank> reverse_start_indices;
    Eigen::array<Eigen::DenseIndex, x_rank> reverse_target_indices;
    for (auto i = 0; i < x_rank; i++) {
      reverse_start_indices[i] = 0;
      reverse_target_indices[i] = 0;
    }
    reverse_start_indices[dim] = 1;
    reverse_target_indices[dim] = input_slice_sizes[dim];

    temp_tensor.slice(reverse_target_indices, reverse_size) =
      temp_tensor.slice(reverse_start_indices, reverse_size).reverse(reverse_dim).conjugate();
  }
  return temp_tensor;
}

template <typename T1, typename T2, int x_rank>
bool ComputeIRFFT(T1 *input_x, T2 *output_y, const std::vector<int64_t> &x_shape, int n, int dim,
                  mindspore::NormMode norm_type) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = x_shape[i];
  }
  Eigen::TensorMap<Eigen::Tensor<T1, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_x[0], tensor_shape);
  Eigen::array<int, 1> dim_array;
  dim_array[0] = dim;
  Eigen::Tensor<T2, x_rank, Eigen::RowMajor> out;

  // irfft
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> complex_out;
  // compute the full fft tensor shape: full_fft_shape[-1] / 2 + 1
  Eigen::array<Eigen::DenseIndex, x_rank> temp_tensor_shape(tensor_shape);
  // check the shape input.shape[dim] cannot be 1
  if (n == 0) {
    MS_EXCEPTION(ValueError) << "For 'IRFFT', the last dimension of the input cannot be 1, but got: "
                             << temp_tensor_shape[dim];
  }
  temp_tensor_shape[dim] = n;

  auto temp_tensor = ReconstructTensor<T1, T2, x_rank>(temp_tensor_shape, in, x_shape, n, dim);
  // do irfft at the last axis:
  complex_out = temp_tensor.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(dim_array);

  out.resize(complex_out.dimensions());
  T1 *complex_out_ptr = complex_out.data();
  for (int i = 0; i < complex_out.size(); i++) {
    *(out.data() + i) = (complex_out_ptr + i)->real();
  }

  double norm_weight = GetNormalizeWeight(n, norm_type);
  T2 *out_ptr = out.data();
  for (int i = 0; i < out.size(); i++) {
    T2 temp_value = *(out_ptr + i);
    temp_value *= norm_weight;
    *(output_y + i) = temp_value;
  }
  return true;
}

template <typename T_in, typename T_mid, typename T_out>
bool IRFFTCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input_ptr = reinterpret_cast<T_in *>(inputs[kIndex0]->device_ptr());
  auto *output_ptr = reinterpret_cast<T_out *>(outputs[kIndex0]->device_ptr());

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * input_element_nums_));
  if (memset_s(calculate_input, sizeof(T_mid) * input_element_nums_, 0, sizeof(T_mid) * input_element_nums_) != EOK) {
    free(calculate_input);
    calculate_input = nullptr;
    MS_LOG(EXCEPTION) << kernel_name_ << " memset_s failed. ";
  }
  GenerateCalculateInput<T_in, T_mid>(input_ptr, calculate_input, input_element_nums_);

  SWITCH_DIM_CALCULATE(T_mid, T_out);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return true;
}

#define IRFFT_CPU_REG(MS_Tin, MS_Tout, T_in, T_mid, T_out) \
  KernelAttr()                                             \
    .AddInputAttr(MS_Tin)                   /* x */        \
    .AddOptionalInputAttr(kNumberTypeInt64) /* n */        \
    .AddInputAttr(kNumberTypeInt64)         /* dim */      \
    .AddOptionalInputAttr(kNumberTypeInt64) /* norm */     \
    .AddOutputAttr(MS_Tout),                               \
    &IRFFTCpuKernelMod::LaunchKernel<T_in, T_mid, T_out>

std::vector<std::pair<KernelAttr, IRFFTCpuKernelMod::IRFFTFunc>> IRFFTCpuKernelMod::func_list_ = {
  {IRFFT_CPU_REG(kNumberTypeInt16, kNumberTypeFloat32, int16_t, complex64, float)},
  {IRFFT_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, int32_t, complex64, float)},
  {IRFFT_CPU_REG(kNumberTypeInt64, kNumberTypeFloat32, int64_t, complex64, float)},
  {IRFFT_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat32, Eigen::half, complex64, float)},
  {IRFFT_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, complex64, float)},
  {IRFFT_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, double, complex128, double)},
  {IRFFT_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat32, complex64, complex64, float)},
  {IRFFT_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat64, complex128, complex128, double)},
};

std::vector<KernelAttr> IRFFTCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, IRFFTFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IRFFT, IRFFTCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
