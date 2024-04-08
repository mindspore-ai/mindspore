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
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "mindspore/core/mindapi/base/types.h"
#include "cpu_kernel/utils/fft_helper.h"

namespace aicpu {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const int kRealFFTSideNum = 2;
const char *const kInputNName = "n";
const char *const kInputNormName = "norm";
const char *kFFT = "FFT";
const char *kIFFT = "IFFT";
const char *kRFFT = "RFFT";

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

#define FFTBASE_COMPUTE_CASE(DTYPE, INTYPE, MIDTYPE, OUTTYPE, CTX)   \
  case (DTYPE): {                                                    \
    uint32_t result = FFTBaseCompute<INTYPE, MIDTYPE, OUTTYPE>(CTX); \
    if (result != KERNEL_STATUS_OK) {                                \
      CUST_KERNEL_LOG_ERROR(ctx, "FFTBase kernel compute failed.");  \
      return result;                                                 \
    }                                                                \
    break;                                                           \
  }

uint32_t FFTBaseCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx.GetOpType());
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           op_name_.c_str());
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
      CUST_KERNEL_LOG_ERROR(ctx, "FFTBase kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

#define EIGEN_FFT_INPUT_RANK_CASE(T1, T2)                                                                         \
  if (x_rank == 1) {                                                                                              \
    EigenFFTBase<T1, T2, 1>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else if (x_rank == 2) {                                                                                       \
    EigenFFTBase<T1, T2, 2>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else if (x_rank == 3) {                                                                                       \
    EigenFFTBase<T1, T2, 3>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else if (x_rank == 4) {                                                                                       \
    EigenFFTBase<T1, T2, 4>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else if (x_rank == 5) {                                                                                       \
    EigenFFTBase<T1, T2, 5>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else if (x_rank == 6) {                                                                                       \
    EigenFFTBase<T1, T2, 6>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else if (x_rank == 7) {                                                                                       \
    EigenFFTBase<T1, T2, 7>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  } else {                                                                                                        \
    EigenFFTBase<T1, T2, 8>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim, rfft_slice); \
  }

template <typename T_in, typename T_out, int x_rank>
bool EigenFFTBase(T_in *input_ptr, T_out *output_ptr, bool forward, double norm_weight,
                  std::vector<int64_t> calculate_shape, int64_t dim, bool rfft_slice) {
  Eigen::array<Eigen::DenseIndex, x_rank> calculate_shape_array;
  for (size_t i = 0; i < x_rank; ++i) {
    calculate_shape_array[i] = calculate_shape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0],
                                                                                     calculate_shape_array);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  std::vector<int32_t> eigem_dim;
  (void)eigem_dim.emplace_back(static_cast<int32_t>(dim));

  if (forward) {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(eigem_dim);
  } else {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(eigem_dim);
  }

  // rfft slice
  if (rfft_slice) {
    auto dims = in.dimensions();
    Eigen::DSizes<Eigen::DenseIndex, x_rank> offsets;
    Eigen::DSizes<Eigen::DenseIndex, x_rank> input_slice_sizes;
    for (auto i = 0; i < x_rank; i++) {
      offsets[i] = 0;
      input_slice_sizes[i] = (i == dim) ? (dims[i] / kRealFFTSideNum + 1) : dims[i];
    }
    Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> slice_out = out.slice(offsets, input_slice_sizes);
    out = slice_out;
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
uint32_t FFTBaseCpuKernel::FFTBaseCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  // step1：Get or set attribute.
  // optional input will be null, we need to adapt the corresponding input index
  int64_t n;
  bool n_is_none =
    (ctx.Input(kFftNIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kFftNIndex)) != kInputNName);
  if (n_is_none) {
    dim_index_--;
    norm_index_--;
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kFftNIndex)->GetData())[0];
  }

  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(dim_index_)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;
  if (n_is_none) {
    n = tensor_shape[dim];
  }

  NormMode norm;
  bool norm_is_none =
    (ctx.Input(norm_index_) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(norm_index_)) != kInputNormName);
  if (norm_is_none) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index_)->GetData())[0]);
  }

  bool forward = IsForwardOp(op_name_);
  double norm_weight = GetNormalized(n, norm, forward);
  bool rfft_slice = (op_name_ == kRFFT);

  // step2：Calculate the required memory based on n and dim.
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element));
  auto ret = memset_s(calculate_input, sizeof(T_mid) * calculate_element, 0, sizeof(T_mid) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, T_mid>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // step4：Run FFT according to parameters
  EIGEN_FFT_INPUT_RANK_CASE(T_mid, T_out);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kRFFT, FFTBaseCpuKernel);
}  // namespace aicpu
