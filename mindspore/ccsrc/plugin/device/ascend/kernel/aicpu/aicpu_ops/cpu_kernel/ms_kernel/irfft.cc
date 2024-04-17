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

#include "cpu_kernel/ms_kernel/irfft.h"
#include <securec.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "mindspore/core/mindapi/base/types.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const uint32_t kIndex0 = 0;
const uint32_t kNIndex = 1;
const uint32_t kDimIndex = 2;
const uint32_t kNormIndex = 3;
const int kRealFFTSideNum = 2;
const char *kIRFFT = "IRFFT";
const std::string op_prefix = "Cust";

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using NormMode = mindspore::NormMode;

#define IRFFT_COMPUTE_CASE(DTYPE, INTYPE, MIDTYPE, OUTTYPE, CTX)   \
  case (DTYPE): {                                                  \
    uint32_t result = IRFFTCompute<INTYPE, MIDTYPE, OUTTYPE>(CTX); \
    if (result != KERNEL_STATUS_OK) {                              \
      CUST_KERNEL_LOG_ERROR(ctx, "IRFFT kernel compute failed.");  \
      return result;                                               \
    }                                                              \
    break;                                                         \
  }

}  // namespace

namespace aicpu {
uint32_t IRFFTCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name = ctx.GetOpType();
  if (op_name.find(op_prefix) == 0) {
    op_name.erase(op_name.begin(), op_name.begin() + op_prefix.size());
  }
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           op_name.c_str());
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    IRFFT_COMPUTE_CASE(DT_INT16, int16_t, complex64, float, ctx)
    IRFFT_COMPUTE_CASE(DT_INT32, int32_t, complex64, float, ctx)
    IRFFT_COMPUTE_CASE(DT_INT64, int64_t, complex64, float, ctx)
    IRFFT_COMPUTE_CASE(DT_FLOAT16, Eigen::half, complex64, float, ctx)
    IRFFT_COMPUTE_CASE(DT_FLOAT, float, complex64, float, ctx)
    IRFFT_COMPUTE_CASE(DT_DOUBLE, double, complex128, double, ctx)
    IRFFT_COMPUTE_CASE(DT_COMPLEX64, complex64, complex64, float, ctx)
    IRFFT_COMPUTE_CASE(DT_COMPLEX128, complex128, complex128, double, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "IRFFT kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

#define SWITCH_DIM_CALCULATE(ctx, T1, T2)                                                  \
  if (x_rank == 1) {                                                                       \
    ComputeIRFFT<T1, T2, 1>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else if (x_rank == 2) {                                                                \
    ComputeIRFFT<T1, T2, 2>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else if (x_rank == 3) {                                                                \
    ComputeIRFFT<T1, T2, 3>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else if (x_rank == 4) {                                                                \
    ComputeIRFFT<T1, T2, 4>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else if (x_rank == 5) {                                                                \
    ComputeIRFFT<T1, T2, 5>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else if (x_rank == 6) {                                                                \
    ComputeIRFFT<T1, T2, 6>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else if (x_rank == 7) {                                                                \
    ComputeIRFFT<T1, T2, 7>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  } else {                                                                                 \
    ComputeIRFFT<T1, T2, 8>(ctx, calculate_input, output_ptr, tensor_shape, n, dim, norm); \
  }

double GetNormalizeWeight(int64_t element_nums, mindspore::NormMode norm_type_) {
  double result = 1.0;
  if (norm_type_ == NormMode::FORWARD) {
    result = 1.0 * element_nums;
  } else if (norm_type_ == NormMode::ORTHO) {
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
bool ComputeIRFFT(CpuKernelContext &ctx, T1 *input_x, T2 *output_y, const std::vector<int64_t> &x_shape, int n, int dim,
                  NormMode norm_type) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = x_shape[i];
  }
  Eigen::TensorMap<Eigen::Tensor<T1, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_x[0], tensor_shape);
  Eigen::array<int, 1> dim_array;
  dim_array[0] = dim;
  Eigen::Tensor<T2, x_rank, Eigen::RowMajor> out;
  std::vector<int64_t> norm_shape(x_shape);

  // irfft
  Eigen::Tensor<T1, x_rank, Eigen::RowMajor> complex_out;
  // compute the full fft tensor shape: full_fft_shape[-1] / 2 + 1
  Eigen::array<Eigen::DenseIndex, x_rank> temp_tensor_shape(tensor_shape);
  // check the shape input.shape[dim] cannot be 1
  if (n == 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "For 'IRFFT' input.shape[dim] cannot be 1 but got [%ld].", temp_tensor_shape[dim]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  temp_tensor_shape[dim] = n;

  auto temp_tensor = ReconstructTensor<T1, T2, x_rank>(temp_tensor_shape, in, x_shape, n, dim);
  // do irfft at the last axis:
  complex_out = temp_tensor.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(dim_array);
  norm_shape.back() = static_cast<int64_t>(temp_tensor_shape.back());

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
uint32_t IRFFTCpuKernel::IRFFTCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  // step1：Get or set attribute.
  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(kDimIndex)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;

  int64_t n;
  if (ctx.Input(kNIndex) == nullptr) {
    n = kRealFFTSideNum * (tensor_shape[dim] - 1);
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kNIndex)->GetData())[0];
  }

  NormMode norm;
  if (ctx.Input(kNormIndex) == nullptr) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(kNormIndex)->GetData())[0]);
  }

  // step2：Calculate the required memory based on n and dim.
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * input_element));
  if (memset_s(calculate_input, sizeof(T_mid) * input_element, 0, sizeof(T_mid) * input_element) != EOK) {
    free(calculate_input);
    calculate_input = nullptr;
    CUST_KERNEL_LOG_ERROR(ctx, "For 'IRFFT', memset_s failed. ");
    return KERNEL_STATUS_INNER_ERROR;
  }
  GenerateCalculateInput<T_in, T_mid>(input_ptr, calculate_input, input_element);

  // step4：Run FFT according to parameters
  SWITCH_DIM_CALCULATE(ctx, T_mid, T_out);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kIRFFT, IRFFTCpuKernel);
}  // namespace aicpu