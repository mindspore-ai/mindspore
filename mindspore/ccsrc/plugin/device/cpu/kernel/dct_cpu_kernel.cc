/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/dct_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "ops/op_utils.h"
#include "kernel/kernel.h"

#define DCT_SWITCH_DIM_CALCULATE(T1, T2, T3)                                               \
  if (x_rank_ == 1) {                                                                      \
    DCTCompute<T1, T2, T3, 1>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else if (x_rank_ == 2) {                                                               \
    DCTCompute<T1, T2, T3, 2>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else if (x_rank_ == 3) {                                                               \
    DCTCompute<T1, T2, T3, 3>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else if (x_rank_ == 4) {                                                               \
    DCTCompute<T1, T2, T3, 4>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else if (x_rank_ == 5) {                                                               \
    DCTCompute<T1, T2, T3, 5>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else if (x_rank_ == 6) {                                                               \
    DCTCompute<T1, T2, T3, 6>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else if (x_rank_ == 7) {                                                               \
    DCTCompute<T1, T2, T3, 7>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  } else {                                                                                 \
    DCTCompute<T1, T2, T3, 8>(p_x, p_y, n_, axis_, norm_type_, x_shape_, forward_, grad_); \
  }

using std::vector;
namespace mindspore {
namespace kernel {
namespace {
static constexpr double M_pi = 3.141592653589793238462643383279;

template <typename T_in, typename T_out>
void ScienceCast(T_in *array_in, T_out *array_out, size_t size) {
  for (size_t i = 0; i < size; i++) {
    if constexpr (std::is_same_v<T_in, T_out>) {
      array_out[i] = static_cast<T_in>(array_in[i]);
    } else if constexpr (std::is_same_v<T_in, std::complex<float>> && std::is_same_v<T_out, float>) {
      array_out[i] = static_cast<T_out>(std::real(array_in[i]));
    } else if constexpr (std::is_same_v<T_in, std::complex<double>> && std::is_same_v<T_out, double>) {
      array_out[i] = static_cast<T_out>(std::real(array_in[i]));
    } else {
      array_out[i] = static_cast<T_out>(array_in[i]);
    }
  }
}

template <int x_rank>
inline Eigen::array<Eigen::DenseIndex, x_rank> GetTensorShape(const std::vector<int64_t> &x_shape) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = x_shape[i];
  }
  return tensor_shape;
}

int64_t get_element_num(const std::vector<int64_t> &shape, size_t rank) {
  size_t back_itr = shape.size();
  int64_t size = 1;
  for (size_t i = 1; i <= rank; ++i) {
    auto dim = shape[back_itr - i];
    size *= dim;
  }
  return size;
}
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
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  x_shape_ = inputs[kIndex0]->GetShapeVector();
  type_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  n_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  axis_ = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  auto norm = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  norm_type_ = static_cast<ops::NormMode>(norm);
  x_rank_ = SizeToLong(x_shape_.size());
  forward_ = inputs[kIndex5]->GetValueWithCheck<bool>();
  grad_ = inputs[kIndex6]->GetValueWithCheck<bool>();

  return KRET_OK;
}

template <typename T>
std::vector<T> TailZeroPadding(T *org_data, const std::vector<int64_t> &x_shape, int64_t n, int axis) {
  // the axis should be canonical and n should be greater than 0
  int64_t element_num = get_element_num(x_shape, x_shape.size());
  if (x_shape[axis] == n) {
    return std::vector<T>(org_data, org_data + element_num);
  }
  std::vector<int64_t> new_shape(x_shape.begin(), x_shape.end());
  new_shape[axis] = n;
  int64_t new_ele_num = get_element_num(new_shape, new_shape.size());
  // zero-initialized
  std::vector<T> new_data(new_ele_num, 0);

  // compute original and new offsets for each axes
  std::vector<int64_t> offsets(x_shape.size(), 0);
  std::vector<int64_t> new_offsets(x_shape.size(), 0);
  for (size_t j = 0; j < x_shape.size(); j++) {
    int64_t pos = SizeToLong(j);
    offsets[j] = std::accumulate(x_shape.begin() + pos + 1, x_shape.end(), 1, std::multiplies<>());
    new_offsets[j] = std::accumulate(new_shape.begin() + pos + 1, new_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < element_num; ++i) {
    std::vector<int64_t> index(x_shape.size(), 0);
    int64_t flat_index = i;
    // compute original coordinates
    for (size_t dim = 0; dim < offsets.size(); ++dim) {
      index[dim] = flat_index / offsets[dim];
      flat_index %= offsets[dim];
    }
    // if n > x.shape[axis] ->truncate, invalid ele should be dropped out
    if (index[axis] >= n) {
      continue;
    }
    int64_t new_flat_index = 0;
    for (size_t dim = 0; dim < new_offsets.size(); ++dim) {
      new_flat_index += index[dim] * new_offsets[dim];
    }
    new_data[new_flat_index] = org_data[i];
  }
  return new_data;
}

// type T should be float/double
template <typename T, int x_rank>
inline Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor> ConstructW4Tensor(int64_t n, int axis) {
  std::vector<std::complex<T>> range(n);
  for (int i = 0; i < n; ++i) {
    range[i] = static_cast<std::complex<T>>(i);
  }
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = i == axis ? n : 1;
  }

  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor>, Eigen::RowMajor> tensor_mp(&range[0],
                                                                                                       tensor_shape);
  Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor> k(tensor_mp);

  auto complex_n = static_cast<std::complex<T>>(n);
  std::complex<T> imag_factor(0, -0.5);
  std::complex<T> pi(M_pi, 0);
  auto coef = imag_factor * pi;
  Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor> revised_k = coef * k / complex_n;

  auto w4 = revised_k.exp();
  return w4;
}

// type T should be complex
template <typename T, int x_rank>
inline Eigen::Tensor<T, x_rank, Eigen::RowMajor> ConstructW4TensorForComplex(int64_t n, int axis) {
  std::vector<T> range(n);
  for (int i = 0; i < n; ++i) {
    range[i] = static_cast<T>(i);
  }
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = i == axis ? n : 1;
  }

  Eigen::TensorMap<Eigen::Tensor<T, x_rank, Eigen::RowMajor>, Eigen::RowMajor> tensor_mp(&range[0], tensor_shape);
  Eigen::Tensor<T, x_rank, Eigen::RowMajor> k(tensor_mp);

  auto complex_n = static_cast<T>(n);
  T imag_factor(0, -0.5);
  T pi(M_pi, 0);
  auto coef = imag_factor * pi;
  Eigen::Tensor<T, x_rank, Eigen::RowMajor> revised_k = coef * k / complex_n;

  auto w4 = revised_k.exp();
  return w4;
}

template <typename T1, int x_rank>
Eigen::Tensor<T1, x_rank, Eigen::RowMajor> Interleave(const Eigen::Tensor<T1, x_rank, Eigen::RowMajor> &in_tensor,
                                                      int axis, const std::vector<int64_t> &x_shape) {
  // construct even part tensor
  Eigen::array<Eigen::DenseIndex, x_rank> strides;
  Eigen::array<Eigen::DenseIndex, x_rank> offsets;
  Eigen::array<Eigen::DenseIndex, x_rank> extends;
  Eigen::array<bool, x_rank> reverse;
  for (int i = 0; i < x_rank; ++i) {
    if (i == axis) {
      strides[i] = 2;
      offsets[i] = 1;
      extends[i] = x_shape[i] - 1;
      reverse[i] = true;
    } else {
      strides[i] = 1;
      offsets[i] = 0;
      extends[i] = x_shape[i];
      reverse[i] = false;
    }
  }
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> even_part = in_tensor.stride(strides);

  // construct odd part tensor
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> sub_tensor = in_tensor.slice(offsets, extends);
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> odd_part = sub_tensor.stride(strides).reverse(reverse);

  // concat
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> out;
  out = even_part.concatenate(odd_part, axis);
  return out;
}

template <typename T_in, typename T_out, int x_rank>
Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> ComputeFFT(const Eigen::Tensor<T_in, x_rank, Eigen::RowMajor> &in_tensor,
                                                         int axis, bool is_inverse) {
  Eigen::array<int, 1> axis_array;
  axis_array[0] = axis;
  // result of fft is complex, depends on T1: float -> complex<float>, double -> complex<double>
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> fft_out;
  if (is_inverse) {
    fft_out = in_tensor.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(axis_array);
  } else {
    fft_out = in_tensor.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axis_array);
  }
  return fft_out;
}

template <typename T, int x_rank>
Eigen::Tensor<T, x_rank, Eigen::RowMajor> OrthoNormalize(const Eigen::Tensor<T, x_rank, Eigen::RowMajor> &real_part,
                                                         const std::vector<int64_t> &shape, int axis) {
  std::vector<T> factor_vec(shape[axis], 2.0);
  factor_vec[0] = 4.0;

  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = i == axis ? shape[i] : 1;
  }

  Eigen::TensorMap<Eigen::Tensor<T, x_rank, Eigen::RowMajor>, Eigen::RowMajor> tensor_mp(&factor_vec[0], tensor_shape);
  Eigen::Tensor<T, x_rank, Eigen::RowMajor> factor_tensor(tensor_mp);
  Eigen::Tensor<T, x_rank, Eigen::RowMajor> out;

  // compute broadcast dimensions for factor_tensor
  Eigen::array<Eigen::DenseIndex, x_rank> bcast;
  for (int i = 0; i < x_rank; ++i) {
    bcast[i] = i == axis ? 1 : shape[i];
  }

  out = real_part / (factor_tensor.broadcast(bcast) * static_cast<T>(shape[axis])).sqrt();
  return out;
}

template <typename T1, typename T2, int x_rank>
Eigen::Tensor<T2, x_rank, Eigen::RowMajor> ForwardCompute(
  Eigen::TensorMap<Eigen::Tensor<T1, x_rank, Eigen::RowMajor>, Eigen::RowMajor> padded_tensor,
  std::vector<int64_t> padded_shape, int axis, ops::NormMode norm_type) {
  // dct_interleave
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> interleaved_tensor =
    Interleave<T1, x_rank>(padded_tensor, axis, padded_shape);

  // compute fft, notice the result of fft is complex
  Eigen::Tensor<T2, x_rank, Eigen::RowMajor> output_real;
  if constexpr (std::is_same<T1, std::complex<float>>::value || std::is_same<T1, std::complex<double>>::value) {
    Eigen::Tensor<T1, x_rank, Eigen::RowMajor> fft_out = ComputeFFT<T1, T1, x_rank>(interleaved_tensor, axis, false);
    Eigen::Tensor<T1, x_rank, Eigen::RowMajor> w4 = ConstructW4TensorForComplex<T1, x_rank>(padded_shape[axis], axis);
    // broadcast w4 tensor
    Eigen::array<Eigen::DenseIndex, x_rank> bcast;
    for (int i = 0; i < x_rank; ++i) {
      bcast[i] = i == axis ? 1 : padded_shape[i];
    }
    Eigen::Tensor<T1, x_rank, Eigen::RowMajor> w4_out = fft_out * w4.broadcast(bcast);
    output_real = 2 * w4_out.real();
  } else {
    Eigen::Tensor<std::complex<T1>, x_rank, Eigen::RowMajor> fft_out =
      ComputeFFT<T1, std::complex<T1>, x_rank>(interleaved_tensor, axis, false);
    Eigen::Tensor<std::complex<T1>, x_rank, Eigen::RowMajor> w4 =
      ConstructW4Tensor<T1, x_rank>(padded_shape[axis], axis);
    // broadcast w4 tensor
    Eigen::array<Eigen::DenseIndex, x_rank> bcast;
    for (int i = 0; i < x_rank; ++i) {
      bcast[i] = i == axis ? 1 : padded_shape[i];
    }
    Eigen::Tensor<std::complex<T1>, x_rank, Eigen::RowMajor> w4_out = fft_out * w4.broadcast(bcast);
    output_real = 2 * w4_out.real();
  }

  if (norm_type == ops::NormMode::ORTHO) {
    auto normalize_out = OrthoNormalize<T2, x_rank>(output_real, padded_shape, axis);
    return normalize_out;
  }
  return output_real;
}

template <typename T, int x_rank>
Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor> PromoteTypeToComplex(
  Eigen::Tensor<T, x_rank, Eigen::RowMajor> in_tensor, Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape) {
  T *in_ptr = in_tensor.data();
  std::vector<std::complex<T>> out_vec(in_ptr, in_ptr + in_tensor.size());
  Eigen::TensorMap<Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor>, Eigen::RowMajor> tensor_mp(&out_vec[0],
                                                                                                       tensor_shape);
  Eigen::Tensor<std::complex<T>, x_rank, Eigen::RowMajor> out(tensor_mp);
  return out;
}

template <typename T1, int x_rank>
Eigen::Tensor<T1, x_rank, Eigen::RowMajor> Deinterleave(const Eigen::Tensor<T1, x_rank, Eigen::RowMajor> &in_tensor,
                                                        int axis, const std::vector<int64_t> &x_shape) {
  // construct offsets and extends
  Eigen::array<Eigen::DenseIndex, x_rank> offsets_even;
  Eigen::array<Eigen::DenseIndex, x_rank> offsets_odd;
  Eigen::array<Eigen::DenseIndex, x_rank> extends_even;
  Eigen::array<Eigen::DenseIndex, x_rank> extends_odd;
  Eigen::array<Eigen::DenseIndex, x_rank> extend_one_step;
  Eigen::array<bool, x_rank> reverse;
  int even_num = ceil(static_cast<float>(x_shape[axis]) / 2.0);
  int odd_num = x_shape[axis] - even_num;
  for (int i = 0; i < x_rank; ++i) {
    offsets_odd[i] = 0;
    offsets_even[i] = 0;
    extends_even[i] = x_shape[i];
    extends_odd[i] = x_shape[i];
    extend_one_step[i] = x_shape[i];
    reverse[i] = false;
  }
  offsets_odd[axis] = even_num;
  extends_even[axis] = even_num;
  extends_odd[axis] = odd_num;
  extend_one_step[axis] = 1;
  reverse[axis] = true;

  // construct odd & even part tensor
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> sub_even = in_tensor.slice(offsets_even, extends_even);
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> sub_odd = in_tensor.slice(offsets_odd, extends_odd).reverse(reverse);

  // concat
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> out;
  for (int i = 0; i < even_num; ++i) {
    offsets_even[axis] = i;
    offsets_odd[axis] = i;
    Eigen::Tensor<T1, x_rank, Eigen::RowMajor> slice = sub_even.slice(offsets_even, extend_one_step);
    Eigen::Tensor<T1, x_rank, Eigen::RowMajor> temp;
    if (i == 0) {
      out = slice;
    } else {
      temp = out.concatenate(slice, axis);
      out = temp;
    }
    if (offsets_odd[axis] >= odd_num) {
      continue;
    }
    slice = sub_odd.slice(offsets_odd, extend_one_step);
    temp = out.concatenate(slice, axis);
    out = temp;
  }
  return out;
}

template <typename T1, typename T2, int x_rank>
Eigen::Tensor<T2, x_rank, Eigen::RowMajor> InverseCompute(
  Eigen::TensorMap<Eigen::Tensor<T1, x_rank, Eigen::RowMajor>, Eigen::RowMajor> padded_tensor,
  std::vector<int64_t> padded_shape, int axis, ops::NormMode norm_type, bool grad) {
  // ortho-normalization
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> norm_out = OrthoNormalize<T1, x_rank>(padded_tensor, padded_shape, axis);
  if (grad && norm_type != ops::NormMode::ORTHO) {
    auto temp_out = OrthoNormalize<T1, x_rank>(norm_out, padded_shape, axis);
    norm_out = temp_out;
  }

  Eigen::Tensor<std::complex<T1>, x_rank, Eigen::RowMajor> w4 = ConstructW4Tensor<T1, x_rank>(padded_shape[axis], axis);

  Eigen::array<Eigen::DenseIndex, x_rank> padded_shape_array = GetTensorShape<x_rank>(padded_shape);
  Eigen::Tensor<std::complex<T1>, x_rank, Eigen::RowMajor> complex_x =
    PromoteTypeToComplex<T1, x_rank>(norm_out, padded_shape_array);

  // broadcast w4 tensor & multiply coefficient
  Eigen::array<Eigen::DenseIndex, x_rank> bcast;
  for (int i = 0; i < x_rank; ++i) {
    bcast[i] = i == axis ? 1 : padded_shape[i];
  }
  std::complex<T1> factor(2, 0);
  std::complex<T1> N(padded_shape[axis], 0);
  auto coef = factor * N;
  Eigen::Tensor<std::complex<T1>, x_rank, Eigen::RowMajor> coef_out = complex_x * coef / w4.broadcast(bcast);

  // compute ifft
  auto ifft_out = ComputeFFT<std::complex<T1>, std::complex<T1>, x_rank>(coef_out, axis, true);
  Eigen::Tensor<T2, x_rank, Eigen::RowMajor> ifft_real = ifft_out.real();

  // de-interleave the result of ifft
  Eigen::Tensor<T2, x_rank, Eigen::RowMajor> out = Deinterleave<T2, x_rank>(ifft_real, axis, padded_shape);
  if (grad && norm_type != ops::NormMode::ORTHO) {
    T2 coeff = static_cast<T2>(2 * padded_shape[axis]);
    auto temp = coeff * out;
    out = temp;
  }
  return out;
}

template <typename T1, typename T2, typename T3, int x_rank>
bool DCTCompute(T1 *input_x, T2 *output_y, int n, int axis, ops::NormMode norm_type,
                const std::vector<int64_t> &x_shape, bool forward, bool grad) {
  // convert data type to float/double
  int64_t element_num = get_element_num(x_shape, x_rank);
  T3 *casted_input = static_cast<T3 *>(malloc(sizeof(T3) * element_num));
  ScienceCast<T1, T3>(input_x, casted_input, element_num);

  // construct eigen tensor
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape = GetTensorShape<x_rank>(x_shape);
  Eigen::TensorMap<Eigen::Tensor<T3, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in_tensor(&casted_input[0],
                                                                                          tensor_shape);

  // canonicalize axis
  axis = axis >= 0 ? axis : axis + x_rank;
  n = n > 0 ? n : x_shape[axis];

  // padding the input tensor
  std::vector<int64_t> padded_shape(x_shape.begin(), x_shape.end());
  padded_shape[axis] = n;
  Eigen::array<Eigen::DenseIndex, x_rank> padded_shape_array = GetTensorShape<x_rank>(padded_shape);

  Eigen::Tensor<T2, x_rank, Eigen::RowMajor> output_real;
  if (forward) {
    std::vector<T3> padded_vector = TailZeroPadding<T3>(casted_input, x_shape, n, axis);
    Eigen::TensorMap<Eigen::Tensor<T3, x_rank, Eigen::RowMajor>, Eigen::RowMajor> padded_tensor(&padded_vector[0],
                                                                                                padded_shape_array);
    output_real = ForwardCompute<T3, T2, x_rank>(padded_tensor, padded_shape, axis, norm_type);
  } else {
    // idct do not take complex input
    if constexpr (std::is_same<T3, std::complex<float>>::value) {
      float *idct_input = static_cast<float *>(malloc(sizeof(float) * element_num));
      ScienceCast<T3, float>(casted_input, idct_input, element_num);
      std::vector<float> padded_vector = TailZeroPadding<float>(idct_input, x_shape, n, axis);
      Eigen::TensorMap<Eigen::Tensor<float, x_rank, Eigen::RowMajor>, Eigen::RowMajor> padded_tensor(
        &padded_vector[0], padded_shape_array);
      output_real = InverseCompute<float, T2, x_rank>(padded_tensor, padded_shape, axis, norm_type, grad);
    } else if constexpr (std::is_same<T3, std::complex<double>>::value) {
      double *idct_input = static_cast<double *>(malloc(sizeof(double) * element_num));
      ScienceCast<T3, double>(casted_input, idct_input, element_num);
      std::vector<double> padded_vector = TailZeroPadding<double>(idct_input, x_shape, n, axis);
      Eigen::TensorMap<Eigen::Tensor<double, x_rank, Eigen::RowMajor>, Eigen::RowMajor> padded_tensor(
        &padded_vector[0], padded_shape_array);
      output_real = InverseCompute<double, T2, x_rank>(padded_tensor, padded_shape, axis, norm_type, grad);
    } else {
      std::vector<T3> padded_vector = TailZeroPadding<T3>(casted_input, x_shape, n, axis);
      Eigen::TensorMap<Eigen::Tensor<T3, x_rank, Eigen::RowMajor>, Eigen::RowMajor> padded_tensor(&padded_vector[0],
                                                                                                  padded_shape_array);
      output_real = InverseCompute<T3, T2, x_rank>(padded_tensor, padded_shape, axis, norm_type, grad);
    }
  }

  T2 *out_ptr = output_real.data();
  for (int i = 0; i < output_real.size(); ++i) {
    *(output_y + i) = *(out_ptr + i);
  }

  free(casted_input);
  return true;
}

template <typename T1, typename T2, typename T3>
bool DCTCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  auto p_x = reinterpret_cast<T1 *>(inputs[kIndex0]->device_ptr());
  auto p_y = reinterpret_cast<T2 *>(outputs[kIndex0]->device_ptr());
  DCT_SWITCH_DIM_CALCULATE(T1, T2, T3);
  return true;
}

#define DCT_CPU_REG(MS_I, MS_O, T_in, I, O, IN)                      \
  KernelAttr()                                                       \
    .AddInputAttr(MS_I)                                /* x */       \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* type */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* n */       \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* axis */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) /* norm */    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* forward */ \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)  /* grad */    \
    .AddOutputAttr(MS_O),                                            \
    &DCTCpuKernelMod::LaunchKernel<I, O, IN>

std::vector<std::pair<KernelAttr, DCTCpuKernelMod::DCTFunc>> DCTCpuKernelMod::func_list_ = {
  // for dct the internal type could be complex
  {DCT_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat32, kNumberTypeComplex64, std::complex<float>, float,
               std::complex<float>)},
  {DCT_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat64, kNumberTypeComplex128, std::complex<double>, double,
               std::complex<double>)},
  {DCT_CPU_REG(kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, float, float, float)},
  {DCT_CPU_REG(kNumberTypeFloat64, kNumberTypeFloat64, kNumberTypeFloat64, double, double, double)},
  // for idct all type should be float/double
  {DCT_CPU_REG(kNumberTypeComplex64, kNumberTypeFloat32, kNumberTypeFloat32, std::complex<float>, float, float)},
  {DCT_CPU_REG(kNumberTypeComplex128, kNumberTypeFloat64, kNumberTypeFloat64, std::complex<double>, double, double)},
  {DCT_CPU_REG(kNumberTypeUInt8, kNumberTypeFloat32, kNumberTypeFloat32, uint8_t, float, float)},
  {DCT_CPU_REG(kNumberTypeUInt16, kNumberTypeFloat32, kNumberTypeFloat32, uint16_t, float, float)},
  {DCT_CPU_REG(kNumberTypeUInt32, kNumberTypeFloat32, kNumberTypeFloat32, uint32_t, float, float)},
  {DCT_CPU_REG(kNumberTypeUInt64, kNumberTypeFloat64, kNumberTypeFloat64, uint64_t, double, double)},
  {DCT_CPU_REG(kNumberTypeInt8, kNumberTypeFloat32, kNumberTypeFloat32, int8_t, float, float)},
  {DCT_CPU_REG(kNumberTypeInt16, kNumberTypeFloat32, kNumberTypeFloat32, int16_t, float, float)},
  {DCT_CPU_REG(kNumberTypeInt32, kNumberTypeFloat32, kNumberTypeFloat32, int32_t, float, float)},
  {DCT_CPU_REG(kNumberTypeInt64, kNumberTypeFloat64, kNumberTypeFloat64, int64_t, double, double)},
  {DCT_CPU_REG(kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32, float16, float, float)},
  {DCT_CPU_REG(kNumberTypeBool, kNumberTypeFloat32, kNumberTypeFloat32, bool, float, float)},
};

std::vector<KernelAttr> DCTCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DCTFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DCT, DCTCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
