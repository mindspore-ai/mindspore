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
#include "dct.h"

#include <iostream>
#include "Eigen/Dense"
#include "context/inc/cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/FFT"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#define DCT_SWITCH_DIM_CALCULATE(T1, T2, T3, x_rank_)                           \
  if (x_rank_ == 1) {                                                           \
    DCTCompute<T1, T2, T3, 1>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else if (x_rank_ == 2) {                                                    \
    DCTCompute<T1, T2, T3, 2>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else if (x_rank_ == 3) {                                                    \
    DCTCompute<T1, T2, T3, 3>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else if (x_rank_ == 4) {                                                    \
    DCTCompute<T1, T2, T3, 4>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else if (x_rank_ == 5) {                                                    \
    DCTCompute<T1, T2, T3, 5>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else if (x_rank_ == 6) {                                                    \
    DCTCompute<T1, T2, T3, 6>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else if (x_rank_ == 7) {                                                    \
    DCTCompute<T1, T2, T3, 7>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  } else {                                                                      \
    DCTCompute<T1, T2, T3, 8>(ctx, n, axis, norm_type, x_shape, forward, grad); \
  }

namespace {
static constexpr int M_interleave = 2;
static constexpr double M_pi = 3.141592653589793238462643383279;
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *kDCT = "DCT";
}  // namespace

namespace aicpu {
uint32_t DCTCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kDCT);
  CUST_KERNEL_HANDLE_ERROR(ctx, DCTCheck(ctx), "[%s] check params failed.", kDCT);

  Tensor *input = ctx.Input(0);
  auto x_shape_ptr = input->GetTensorShape();
  int64_t x_rank = x_shape_ptr->GetDims();
  std::vector<int64_t> x_shape = x_shape_ptr->GetDimSizes();
  // get attr values
  AttrValue *attr2 = ctx.GetAttr("n");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr2, KERNEL_STATUS_PARAM_INVALID, "Get param[n] failed.")
  uint64_t n = attr2->GetInt();
  AttrValue *attr3 = ctx.GetAttr("axis");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr3, KERNEL_STATUS_PARAM_INVALID, "Get param[axis] failed.")
  uint64_t axis = attr3->GetInt();
  AttrValue *attr4 = ctx.GetAttr("norm");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr4, KERNEL_STATUS_PARAM_INVALID, "Get param[norm] failed.")
  uint64_t norm_type = attr4->GetInt();
  AttrValue *attr5 = ctx.GetAttr("forward");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr5, KERNEL_STATUS_PARAM_INVALID, "Get param[forward] failed.")
  bool forward = attr5->GetBool();
  AttrValue *attr6 = ctx.GetAttr("grad");
  CUST_KERNEL_CHECK_NULLPTR(ctx, attr6, KERNEL_STATUS_PARAM_INVALID, "Get param[grad] failed.")
  bool grad = attr6->GetBool();

  auto input_type = ctx.Input(0)->GetDataType();
  if (input_type == DT_COMPLEX128) {
    if (forward) {
      DCT_SWITCH_DIM_CALCULATE(std::complex<double>, double, std::complex<double>, x_rank)
    } else {
      DCT_SWITCH_DIM_CALCULATE(std::complex<double>, double, double, x_rank)
    }
  } else if (input_type == DT_COMPLEX64) {
    if (forward) {
      DCT_SWITCH_DIM_CALCULATE(std::complex<float>, float, std::complex<float>, x_rank)
    } else {
      DCT_SWITCH_DIM_CALCULATE(std::complex<float>, float, float, x_rank)
    }
  } else if (input_type == DT_DOUBLE) {
    DCT_SWITCH_DIM_CALCULATE(double, double, double, x_rank)
  } else if (input_type == DT_FLOAT) {
    DCT_SWITCH_DIM_CALCULATE(float, float, float, x_rank)
  } else if (input_type == DT_FLOAT16) {
    DCT_SWITCH_DIM_CALCULATE(Eigen::half, float, float, x_rank)
  } else if (input_type == DT_INT8) {
    DCT_SWITCH_DIM_CALCULATE(int8_t, float, float, x_rank)
  } else if (input_type == DT_INT16) {
    DCT_SWITCH_DIM_CALCULATE(int16_t, float, float, x_rank)
  } else if (input_type == DT_INT32) {
    DCT_SWITCH_DIM_CALCULATE(int32_t, float, float, x_rank)
  } else if (input_type == DT_INT64) {
    DCT_SWITCH_DIM_CALCULATE(int64_t, double, double, x_rank)
  } else if (input_type == DT_UINT8) {
    DCT_SWITCH_DIM_CALCULATE(uint8_t, float, float, x_rank)
  } else if (input_type == DT_UINT16) {
    DCT_SWITCH_DIM_CALCULATE(uint16_t, float, float, x_rank)
  } else if (input_type == DT_UINT32) {
    DCT_SWITCH_DIM_CALCULATE(uint32_t, float, float, x_rank)
  } else if (input_type == DT_UINT64) {
    DCT_SWITCH_DIM_CALCULATE(uint64_t, double, double, x_rank)
  } else if (input_type == DT_BOOL) {
    DCT_SWITCH_DIM_CALCULATE(bool, float, float, x_rank)
  } else {
    CUST_KERNEL_LOG_ERROR(ctx, "DCT kernel data type [%s] not support.", DTypeStr(input_type).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t DCTCpuKernel::DCTCheck(CpuKernelContext &ctx) {
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output data failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get input tensor shape failed.")
  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.Output(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID,
                            "Get output tensor shape failed.")

  return KERNEL_STATUS_OK;
}

namespace {
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
    int64_t pos = static_cast<int64_t>(j);
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

template <typename T1, int x_rank>
Eigen::Tensor<T1, x_rank, Eigen::RowMajor> Interleave(const Eigen::Tensor<T1, x_rank, Eigen::RowMajor> &in_tensor,
                                                      int axis, const std::vector<int64_t> &x_shape) {
  if (in_tensor.dimension(axis) < M_interleave) {
    return in_tensor;
  }
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
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> out = even_part.concatenate(odd_part, axis);
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
  std::vector<int64_t> padded_shape, int axis, int norm_type) {
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

  if (norm_type == 2) {
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
  if (in_tensor.dimension(axis) < M_interleave) {
    return in_tensor;
  }
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
  std::vector<int64_t> padded_shape, int axis, int norm_type, bool grad) {
  // ortho-normalization
  constexpr int kTwo = 2;
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> norm_out;
  if (norm_type == kTwo) {
    norm_out = OrthoNormalize<T1, x_rank>(padded_tensor, padded_shape, axis);
  } else {
    norm_out = padded_tensor;
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
  return out;
}

}  // namespace

template <typename T1, typename T2, typename T3, int x_rank>
uint32_t DCTCpuKernel::DCTCompute(CpuKernelContext &ctx, int n, int axis, int norm_type,
                                  const std::vector<int64_t> &x_shape, bool forward, bool grad) {
  auto input_x = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());

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
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kDCT, DCTCpuKernel);
}  // namespace aicpu
