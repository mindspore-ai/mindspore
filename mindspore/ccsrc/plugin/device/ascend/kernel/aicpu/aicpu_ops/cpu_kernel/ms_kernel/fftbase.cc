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

#include "cpu_kernel/ms_kernel/fftbase.h"
#include <securec.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "mindspore/core/mindapi/base/types.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kNIndex = 1;
const uint32_t kDimIndex = 2;
const uint32_t kNormIndex = 3;
const char *kFFT = "FFT";
const char *kIFFT = "IFFT";
const std::string op_prefix = "Cust";
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using NormMode = mindspore::NormMode;

#define FFTBASE_COMPUTE_CASE(DTYPE, INTYPE, MIDTYPE, OUTTYPE, CTX)   \
  case (DTYPE): {                                                    \
    uint32_t result = FFTBaseCompute<INTYPE, MIDTYPE, OUTTYPE>(CTX); \
    if (result != KERNEL_STATUS_OK) {                                \
      KERNEL_LOG_ERROR("FFTBase kernel compute failed.");            \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

}  // namespace

namespace aicpu {
uint32_t FFTBaseCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name = ctx.GetOpType();
  if (op_name.find(op_prefix) == 0) {
    op_name.erase(op_name.begin(), op_name.begin() + op_prefix.size());
  }
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", op_name.c_str());
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    FFTBASE_COMPUTE_CASE(DT_INT16, int16_t, float, complex64, ctx)
    FFTBASE_COMPUTE_CASE(DT_INT32, int32_t, float, complex64, ctx)
    FFTBASE_COMPUTE_CASE(DT_INT64, int64_t, float, complex64, ctx)
    FFTBASE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, float, complex64, ctx)
    FFTBASE_COMPUTE_CASE(DT_FLOAT, float, float, complex64, ctx)
    FFTBASE_COMPUTE_CASE(DT_DOUBLE, double, double, complex128, ctx)
    FFTBASE_COMPUTE_CASE(DT_COMPLEX64, complex64, complex64, complex64, ctx)
    FFTBASE_COMPUTE_CASE(DT_COMPLEX128, complex128, complex128, complex128, ctx)
    default:
      KERNEL_LOG_ERROR("FFTBase kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

#define SWITCH_DIM_CALCULATE(T1, T2)                                                                                   \
  if (x_rank == 1) {                                                                                                   \
    ComputeFFTBase<T1, T2, 1>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else if (x_rank == 2) {                                                                                            \
    ComputeFFTBase<T1, T2, 2>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else if (x_rank == 3) {                                                                                            \
    ComputeFFTBase<T1, T2, 3>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else if (x_rank == 4) {                                                                                            \
    ComputeFFTBase<T1, T2, 4>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else if (x_rank == 5) {                                                                                            \
    ComputeFFTBase<T1, T2, 5>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else if (x_rank == 6) {                                                                                            \
    ComputeFFTBase<T1, T2, 6>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else if (x_rank == 7) {                                                                                            \
    ComputeFFTBase<T1, T2, 7>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  } else {                                                                                                             \
    ComputeFFTBase<T1, T2, 8>(calculate_input, output_ptr, forward, norm, n, dim, calculate_shape, calculate_element); \
  }

double Getnormalized(int64_t element_nums_, NormMode norm, bool forward) {
  double result = 1.0;
  if (forward) {
    if (norm == NormMode::FORWARD) {
      result = 1.0 / element_nums_;
    }
    if (norm == NormMode::ORTHO) {
      result = 1.0 / sqrt(static_cast<double>(element_nums_));
    }
  }
  if (!forward) {
    if (norm == NormMode::FORWARD) {
      result = 1.0 * element_nums_;
    }
    if (norm == NormMode::ORTHO) {
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
bool ComputeFFTBase(T_in *input_ptr, T_out *output_ptr, bool forward, NormMode norm, int64_t n, int64_t dim,
                    std::vector<int64_t> &tensor_shape, int64_t element_nums_) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape_array;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape_array[i] = tensor_shape[i];
  }
  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0], tensor_shape_array);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  Eigen::array<int, 1> dims_array;
  dims_array[0] = dim;

  if (forward) {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(dims_array);
  } else {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(dims_array);
  }

  int64_t fft_element_nums_ = tensor_shape[dim];
  double norm_weight = Getnormalized(fft_element_nums_, norm, forward);

  T_out *out_ptr = out.data();
  for (int i = 0; i < out.size(); i++) {
    T_out temp_value = *(out_ptr + i);
    temp_value *= norm_weight;
    *(output_ptr + i) = temp_value;
  }

  return true;
}

template <typename T_in, typename T_mid, typename T_out>
uint32_t FFTBaseCpuKernel::FFTBaseCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  // step1：Get or set attribute.
  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(kDimIndex)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;

  int64_t n;
  if (ctx.Input(kNIndex) == nullptr) {
    n = tensor_shape[dim];
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kNIndex)->GetData())[0];
  }

  NormMode norm;
  if (ctx.Input(kNormIndex) == nullptr) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(kNormIndex)->GetData())[0]);
  }

  bool forward = (op_name == kFFT);

  // step2：Calculate the required memory based on n and dim.
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element));
  memset_s(calculate_input, sizeof(T_mid) * calculate_element, 0, sizeof(T_mid) * calculate_element);
  GenarateCalculateInput<T_in, T_mid>(input_ptr, calculate_input, input_element, tensor_shape, calculate_shape, n, dim);

  // step4：Run FFT according to parameters
  SWITCH_DIM_CALCULATE(T_mid, T_out);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFT, FFTBaseCpuKernel);
}  // namespace aicpu