/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "fft_with_size.h"
#include <iostream>
#include "Eigen/Dense"
#include "context/inc/cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/FFT"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#define FFTWITHSIZE_CALCULATE_TYPE(type1, type2, dim, real, rinverse) \
  return FFTWithSizeCompute<type1, type2, dim, real, rinverse>(ctx, onesided, inverse, normalized, checked_signal_size);

#define FFTWITHSIZE_SWITCH_DIM_CALCULATE(type1, type2, real, rinverse, signal_ndim) \
  if (signal_ndim == 1) {                                                           \
    FFTWITHSIZE_CALCULATE_TYPE(type1, type2, 1, real, rinverse)                     \
  } else if (signal_ndim == 2) {                                                    \
    FFTWITHSIZE_CALCULATE_TYPE(type1, type2, 2, real, rinverse)                     \
  } else {                                                                          \
    FFTWITHSIZE_CALCULATE_TYPE(type1, type2, 3, real, rinverse)                     \
  }

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const int64_t kNum2 = 2;
const char *kFFTWithSize = "FFTWithSize";
}  // namespace

namespace aicpu {
uint32_t FFTWithSizeCpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "Check FFTWithSize params failed.");
  Tensor *input = ctx.Input(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input, KERNEL_STATUS_INNER_ERROR, "[%s] get input[%u] failed.",
                            ctx.GetOpType().c_str(), 0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input0 data failed.");
  auto input_x_Shape = input->GetTensorShape();
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_x_Shape, KERNEL_STATUS_PARAM_INVALID, "Get input_x_Shape failed.")
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "FFTWithSizeCpuKernel[%s] , input_0: size[%llu] "
                        "output: size[%llu].",
                        ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
  AttrValue *attr1 = ctx.GetAttr("signal_ndim");
  uint64_t signal_ndim = attr1->GetInt();
  AttrValue *attr2 = ctx.GetAttr("inverse");
  bool inverse = attr2->GetBool();
  AttrValue *attr = ctx.GetAttr("signal_sizes");
  std::vector<int64_t> checked_signal_size = attr ? attr->GetListInt() : std::vector<int64_t>();
  AttrValue *attr3 = ctx.GetAttr("norm");
  std::string normalized = attr3->GetString();
  AttrValue *attr4 = ctx.GetAttr("onesided");
  bool onesided = attr4->GetBool();

  // error detect
  auto x_shape = input->GetTensorShape();
  uint32_t x_dims = x_shape->GetDims();
  if (signal_ndim > 3 || signal_ndim < 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "signal_ndim should less than 4 and greater than 0");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (signal_ndim > x_dims) {
    CUST_KERNEL_LOG_ERROR(ctx, "Input must have rank at least [%d] but got:[%d]", signal_ndim, x_dims);
    return KERNEL_STATUS_PARAM_INVALID;
  }

  auto input_type = ctx.Input(0)->GetDataType();
  auto output_type = ctx.Output(0)->GetDataType();
  switch (input_type) {
    case DT_DOUBLE:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(double, std::complex<double>, true, false, signal_ndim);  // rfft
      break;
    case DT_FLOAT:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(float, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_UINT8:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(uint8_t, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_INT8:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(int8_t, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_INT16:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(int16_t, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_INT32:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(int32_t, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_INT64:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(int64_t, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_BOOL:
      FFTWITHSIZE_SWITCH_DIM_CALCULATE(bool, std::complex<float>, true, false, signal_ndim);  // rfft
      break;
    case DT_COMPLEX128:
      if (output_type == DT_COMPLEX128) {
        FFTWITHSIZE_SWITCH_DIM_CALCULATE(std::complex<double>, std::complex<double>, false, false,
                                         signal_ndim);  // fft ifft
      } else {
        FFTWITHSIZE_SWITCH_DIM_CALCULATE(std::complex<double>, double, true, true, signal_ndim);  // irfft
      }
      break;
    case DT_COMPLEX64:
      if (output_type == DT_COMPLEX64) {
        FFTWITHSIZE_SWITCH_DIM_CALCULATE(std::complex<float>, std::complex<float>, false, false,
                                         signal_ndim);  // fft ifft
      } else {
        FFTWITHSIZE_SWITCH_DIM_CALCULATE(std::complex<float>, float, true, true, signal_ndim);  // irfft
      }
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "FFTWithSize kernel data type [%s] not support.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

double FFTWithSizeCpuKernel::Getnormalized(std::vector<int64_t> shape, std::string normalized, bool is_reverse,
                                           uint64_t dim, bool is_real, std::vector<int64_t> end_shape) {
  double result{0};
  int64_t n = 1;
  int64_t m = 1;
  for (size_t i = shape.size() - 1, j = dim; j > 0; i--, j--) {
    n *= shape[i];
  }
  for (size_t i = end_shape.size() - 1, j = dim; j > 0; i--, j--) {
    m *= end_shape[i];
  }
  if (!is_reverse) {
    if (normalized == "forward") result = 1.0 / n;
    if (normalized == "backward") result = 1.0;
    if (normalized == "ortho") result = 1.0 / sqrt((double)n);
  }
  if (is_reverse) {
    if (is_real) {
      if (normalized == "forward") result = 1.0 * m;
      if (normalized == "backward") result = 1.0;
      if (normalized == "ortho") result = 1.0 * sqrt((double)m);
    } else {
      if (normalized == "forward") result = 1.0 * n;
      if (normalized == "backward") result = 1.0;
      if (normalized == "ortho") result = 1.0 * sqrt((double)n);
    }
  }
  return result;
}

namespace {
template <int signal_ndim>
inline Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> GetFlatShape(std::vector<int64_t> &x_shape, int x_dims) {
  Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> tensor_shape;
  if (x_dims == signal_ndim) {
    tensor_shape[0] = 1;
    for (int i = 0; i < x_dims; i++) {
      tensor_shape[i + 1] = x_shape[i];
    }
  } else if (x_dims == signal_ndim + 1) {
    for (int i = 0; i < x_dims; i++) {
      tensor_shape[i] = x_shape[i];
    }
  } else if (x_dims > signal_ndim + 1) {
    tensor_shape[0] = 1;
    for (int i = 0; i < x_dims - signal_ndim; i++) {
      tensor_shape[0] *= x_shape[i];
    }
    for (int j = x_dims - signal_ndim, i = 1; j < x_dims; j++, i++) {
      tensor_shape[i] = x_shape[j];
    }
  }
  return tensor_shape;
}

template <unsigned int size, unsigned int from, unsigned int to>
inline void change_axes(CpuKernelContext &ctx, Eigen::array<unsigned int, size> *axes) {
  for (unsigned i = from; i <= (unsigned)to; i++) {
    axes->operator[](i - 1) = i;
  }
  return;
}

template <typename T1, typename T2, int signal_ndim, bool is_real, bool real_inverse>
class FFTInnerComputer {
 public:
  uint32_t compute(CpuKernelContext &ctx, bool onesided,
                   Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> &tensor_shape,
                   Eigen::TensorMap<Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor>, Eigen::RowMajor> &in,
                   Eigen::array<unsigned int, signal_ndim> &axes,
                   Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> &out,
                   std::vector<int64_t> &checked_signal_size) {
    CUST_KERNEL_LOG_ERROR(ctx, "FFTWithSize kernel Inner Error");
    return KERNEL_STATUS_PARAM_INVALID;
  }
};

// class template partial specializations
template <typename T1, typename T2, int signal_ndim>
class FFTInnerComputer<T1, T2, signal_ndim, true, true> {
 public:
  uint32_t compute(  // irfft 1d-3d
    CpuKernelContext &ctx, const bool onesided, const bool inverse, const T1 *input_x,
    Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> &tensor_shape,
    Eigen::TensorMap<Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor>, Eigen::RowMajor> &in,
    Eigen::array<unsigned int, signal_ndim> &axes, Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> &out,
    std::vector<int64_t> &checked_signal_size) {
    Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor> complex_out;
    if (onesided) {
      // compute the full fft tensor shape: full_fft_shape[-1] / 2 + 1
      Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> temp_tensor_shape(tensor_shape);
      if (checked_signal_size.empty()) {
        temp_tensor_shape[signal_ndim] = (temp_tensor_shape[signal_ndim] - 1) * kNum2;
      } else {
        if (checked_signal_size.back() / kNum2 + 1 == temp_tensor_shape[signal_ndim]) {
          temp_tensor_shape[signal_ndim] = checked_signal_size.back();
        } else {
          CUST_KERNEL_LOG_ERROR(ctx,
                                "FFTWithSize kernel IRFFT checked_signal_size [%s] not "
                                "support.",
                                VectorToString(checked_signal_size).c_str());
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
      if (temp_tensor_shape.back() == tensor_shape.back()) {
        // fake oneside, such as (3,2) -> (3,2) {2/2+1 == 2}
        // thus, there is no need to reconstruct signal tensor
        complex_out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(axes);
      } else {
        // Reconstruct the full fft tensor: temp_tensor
        Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor> temp_tensor(temp_tensor_shape);
        temp_tensor.setZero();
        Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> zero_offsets;
        Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> input_slice_sizes(in.dimensions());
        temp_tensor.slice(zero_offsets, input_slice_sizes) = in;

        // do ifft at outer axes, then the data is symmetrical on the last axis
        if (signal_ndim > 1) {
          Eigen::array<unsigned int, signal_ndim - 1> outer_axes;
          change_axes<signal_ndim - 1, 1, signal_ndim - 1>(ctx, &outer_axes);
          temp_tensor = temp_tensor.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(outer_axes);
        }

        // rebuild the last axis with symmetrical data
        Eigen::array<bool, signal_ndim + 1> reverse_last_axis;
        for (auto i = 0; i <= signal_ndim; i++) {
          reverse_last_axis[i] = i == signal_ndim;
        }
        auto reverse_size = input_slice_sizes;
        reverse_size[signal_ndim] = temp_tensor_shape[signal_ndim] - input_slice_sizes[signal_ndim];
        Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> reverse_start_indices;
        reverse_start_indices[signal_ndim] = 1;
        Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> reverse_target_indices;
        reverse_target_indices[signal_ndim] = input_slice_sizes[signal_ndim];
        temp_tensor.slice(reverse_target_indices, reverse_size) =
          temp_tensor.slice(reverse_start_indices, reverse_size).reverse(reverse_last_axis).conjugate();

        // do irfft at the last axis:
        auto inner_axis = Eigen::array<unsigned int, 1>{signal_ndim};
        complex_out = temp_tensor.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(inner_axis);
      }
    } else {
      complex_out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(axes);
    }
    out.resize(complex_out.dimensions());
    T1 *complex_out_ptr = complex_out.data();
    for (int i = 0; i < complex_out.size(); i++) {
      *(out.data() + i) = (complex_out_ptr + i)->real();
    }
    return KERNEL_STATUS_OK;
  }
};

template <typename T1, typename T2, int signal_ndim>
class FFTInnerComputer<T1, T2, signal_ndim, true, false> {
 public:
  uint32_t compute(  // rfft 1d-3d
    CpuKernelContext &ctx, const bool onesided, const bool inverse, const T1 *input_x,
    Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> &tensor_shape,
    Eigen::TensorMap<Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor>, Eigen::RowMajor> &in,
    Eigen::array<unsigned int, signal_ndim> &axes, Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> &out,
    std::vector<int64_t> &checked_signal_size) {
    Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> complex_in(in.dimensions());
    T2 *in_data_ptr = complex_in.data();
    for (int i = 0; i < in.size(); i++) {
      (in_data_ptr + i)->real(*(input_x + i));
      (in_data_ptr + i)->imag(0);
    }
    Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> full_fft =
      complex_in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);
    if (onesided) {
      auto dims = in.dimensions();
      Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> offsets;
      Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> input_slice_sizes;
      for (auto i = 0; i <= signal_ndim; i++) {
        input_slice_sizes[i] = (i == signal_ndim) ? (dims[i] / kNum2 + 1) : dims[i];
      }
      out = full_fft.slice(offsets, input_slice_sizes);
    } else {
      out = full_fft;
    }
    return KERNEL_STATUS_OK;
  }
};

template <typename T1, typename T2, int signal_ndim, bool real_inverse>
class FFTInnerComputer<T1, T2, signal_ndim, false, real_inverse> {
 public:
  uint32_t compute(  // fft and ifft 1d-3d
    CpuKernelContext &ctx, const bool onesided, const bool inverse, const T1 *input_x,
    Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> &tensor_shape,
    Eigen::TensorMap<Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor>, Eigen::RowMajor> &in,
    Eigen::array<unsigned int, signal_ndim> &axes, Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> &out,
    std::vector<int64_t> &checked_signal_size) {
    if (inverse) {
      out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(axes);
    } else {
      out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(axes);
    }
    return KERNEL_STATUS_OK;
  }
};
}  // namespace

template <typename T1, typename T2, int signal_ndim, bool is_real, bool real_inverse>
uint32_t FFTWithSizeCpuKernel::FFTWithSizeCompute(CpuKernelContext &ctx, bool onesided, bool inverse,
                                                  std::string normalized, std::vector<int64_t> &checked_signal_size) {
  auto input_x = reinterpret_cast<T1 *>(ctx.Input(0)->GetData());
  auto output_y = reinterpret_cast<T2 *>(ctx.Output(0)->GetData());
  auto x_shape_ptr = ctx.Input(0)->GetTensorShape();
  auto y_shape_ptr = ctx.Output(0)->GetTensorShape();
  int32_t x_dims = x_shape_ptr->GetDims();
  std::vector<int64_t> x_shape = x_shape_ptr->GetDimSizes();
  std::vector<int64_t> y_shape = x_shape;
  Eigen::DSizes<Eigen::DenseIndex, signal_ndim + 1> tensor_shape = GetFlatShape<signal_ndim>(x_shape, x_dims);
  Eigen::TensorMap<Eigen::Tensor<T1, signal_ndim + 1, Eigen::RowMajor>, Eigen::RowMajor> in(&input_x[0], tensor_shape);
  Eigen::array<unsigned int, signal_ndim> axes;
  change_axes<signal_ndim, 1, signal_ndim>(ctx, &axes);
  Eigen::Tensor<T2, signal_ndim + 1, Eigen::RowMajor> out;
  FFTInnerComputer<T1, T2, signal_ndim, is_real, real_inverse> inner_computer;
  uint32_t status_code =
    inner_computer.compute(ctx, onesided, inverse, input_x, tensor_shape, in, axes, out, checked_signal_size);
  if (status_code != KERNEL_STATUS_OK) {
    return status_code;
  }
  if (is_real) {
    inverse = real_inverse;
  }
  auto end_shape = y_shape;
  end_shape.back() = out.dimensions().back();
  auto norm = Getnormalized(x_shape, normalized, inverse, signal_ndim, is_real, end_shape);
  out = norm * out;
  T2 *out_ptr = out.data();
  auto out_count = out.size();
  std::copy(out_ptr, out_ptr + out_count, output_y);
  y_shape.back() = out.dimensions().back();
  y_shape_ptr->SetDimSizes(y_shape);
  CUST_KERNEL_LOG_DEBUG(ctx,
                        "FFTWithSizeCpuKernel[%s] after, input_0: size[%llu] "
                        "output: size[%llu].",
                        ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Output(0)->GetDataSize());
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kFFTWithSize, FFTWithSizeCpuKernel);
}  // namespace aicpu
