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
#include "common/kernel_errcode.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "mindspore/core/mindapi/base/types.h"
#include "cpu_kernel/utils/fft_helper.h"
#include "inc/kernel_log.h"
#include "base/bfloat16.h"
#include <vector>
#include <securec.h>

namespace aicpu {
const char *const kInputNName = "n";
const char *const kInputNormName = "norm";
const char *kFFT = "FFT";
const char *kIFFT = "IFFT";
const char *kRFFT = "RFFT";
const char *kIRFFT = "IRFFT";
const char *kHFFT = "HFFT";
const char *kIHFFT = "IHFFT";

constexpr int kOnesideDivisor = 2;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

uint32_t FFTBaseCpuKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto output_tensor = ctx.Output(0);
  output_type_ = output_tensor->GetDataType();
  auto input_tensor = ctx.Input(0);
  input_type_ = input_tensor->GetDataType();
  return kAicpuKernelStateSucess;
}

template <typename T_in, typename T_out>
uint32_t FFTBaseCpuKernel::FFTBaseComputeR2C(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<std::complex<T_out> *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  std::size_t dim_index = kFftDimIndex;
  std::size_t norm_index = kFftNormIndex;
  auto op_name = GetOpName(ctx);

  int64_t n;
  bool n_is_none =
    (ctx.Input(kFftNIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kFftNIndex)) != kInputNName);
  if (n_is_none) {
    dim_index--;
    norm_index--;
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kFftNIndex)->GetData())[0];
  }

  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(dim_index)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;
  if (n_is_none) {
    n = tensor_shape[dim];
  }

  NormMode norm;
  bool norm_is_none =
    (ctx.Input(norm_index) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(norm_index)) != kInputNormName);
  if (norm_is_none) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index)->GetData())[0]);
  }

  bool forward = IsForwardOp(op_name);
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;
  double norm_weight = GetNormalized(n, norm, forward);
  T_out fct = static_cast<T_out>(norm_weight);

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element));
  auto ret = memset_s(calculate_input, sizeof(T_out) * calculate_element, 0, sizeof(T_out) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // step4：Run FFT according to parameters
  std::vector<int64_t> dim_vec(1, dim);
  forward = op_name == kIHFFT ? !forward : forward;
  PocketFFTR2C<T_out>(calculate_input, output_ptr, forward, fct, calculate_shape, dim_vec);

  if (op_name == kHFFT || op_name == kIHFFT) {
    std::transform(output_ptr, output_ptr + ctx.Output(kIndex0)->NumElements(), output_ptr,
                   [](std::complex<T_out> x) { return std::conj(x); });
  }
  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
};

template <typename T_in, typename T_out>
uint32_t FFTBaseCpuKernel::FFTBaseComputeC2R(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  std::size_t dim_index = kFftDimIndex;
  std::size_t norm_index = kFftNormIndex;
  auto op_name = GetOpName(ctx);

  // step1：optional input will be null, we need to adapt the corresponding input index
  int64_t n;
  bool n_is_none =
    (ctx.Input(kFftNIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kFftNIndex)) != kInputNName);
  if (n_is_none) {
    dim_index--;
    norm_index--;
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kFftNIndex)->GetData())[0];
  }

  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(dim_index)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;
  if (n_is_none) {
    if (op_name == kHFFT || op_name == kIRFFT) {
      n = (tensor_shape[dim] - 1) * kOnesideDivisor;
    } else {
      n = tensor_shape[dim];
    }
  }

  NormMode norm;
  bool norm_is_none =
    (ctx.Input(norm_index) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(norm_index)) != kInputNormName);
  if (norm_is_none) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index)->GetData())[0]);
  }

  bool forward = IsForwardOp(op_name);
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;
  double norm_weight = GetNormalized(n, norm, forward);
  T_out fct = static_cast<T_out>(norm_weight);

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  std::complex<T_out> *calculate_input =
    static_cast<std::complex<T_out> *>(malloc(sizeof(std::complex<T_out>) * calculate_element));
  auto ret = memset_s(calculate_input, sizeof(std::complex<T_out>) * calculate_element, 0,
                      sizeof(std::complex<T_out>) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  if (op_name == kHFFT) {
    std::transform(calculate_input, calculate_input + calculate_element, calculate_input,
                   [](std::complex<T_out> x) { return std::conj(x); });
    forward = !forward;
  }
  // step4：Run FFT according to parameters
  std::vector<int64_t> dim_vec(1, dim);
  PocketFFTC2R<T_out>(calculate_input, output_ptr, forward, fct, calculate_shape, dim_vec);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
};

template <typename T_in, typename T_out>
uint32_t FFTBaseCpuKernel::FFTBaseComputeC2C(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<std::complex<T_out> *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  std::size_t dim_index = kFftDimIndex;
  std::size_t norm_index = kFftNormIndex;
  auto op_name = GetOpName(ctx);

  int64_t n;
  bool n_is_none =
    (ctx.Input(kFftNIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kFftNIndex)) != kInputNName);
  if (n_is_none) {
    dim_index--;
    norm_index--;
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kFftNIndex)->GetData())[0];
  }

  int64_t dim = reinterpret_cast<int64_t *>(ctx.Input(dim_index)->GetData())[0];
  dim = dim < 0 ? x_rank + dim : dim;
  if (n_is_none) {
    n = op_name == kHFFT ? (tensor_shape[dim] - 1) * 2 : tensor_shape[dim];
  }

  NormMode norm;
  bool norm_is_none =
    (ctx.Input(norm_index) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(norm_index)) != kInputNormName);
  if (norm_is_none) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index)->GetData())[0]);
  }

  bool forward = IsForwardOp(op_name);
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;
  double norm_weight = GetNormalized(n, norm, forward);
  T_out fct = static_cast<T_out>(norm_weight);

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  std::complex<T_out> *calculate_input =
    static_cast<std::complex<T_out> *>(malloc(sizeof(std::complex<T_out>) * calculate_element));
  auto ret = memset_s(calculate_input, sizeof(std::complex<T_out>) * calculate_element, 0,
                      sizeof(std::complex<T_out>) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // step4：Run FFT according to parameters
  std::vector<int64_t> dim_vec(1, dim);
  PocketFFTC2C<T_out>(calculate_input, output_ptr, forward, fct, calculate_shape, dim_vec);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
};

template <typename T_in, typename T_out>
uint32_t FFTBaseCpuKernel::FFTBaseCompute(CpuKernelContext &ctx) {
  auto op_name = GetOpName(ctx);
  if (op_name == kRFFT || op_name == kIHFFT) {
    FFTBaseCpuKernel::FFTBaseComputeR2C<T_in, T_out>(ctx);
  }
  if (op_name == kIRFFT || op_name == kHFFT) {
    FFTBaseCpuKernel::FFTBaseComputeC2R<T_in, T_out>(ctx);
  }
  if (op_name == kFFT || op_name == kIFFT) {
    FFTBaseCpuKernel::FFTBaseComputeC2C<T_in, T_out>(ctx);
  }
  return KERNEL_STATUS_OK;
};

uint32_t FFTBaseCpuKernel::Compute(CpuKernelContext &ctx) {
  ParseKernelParam(ctx);
  std::map<int, std::map<int, std::function<uint32_t(CpuKernelContext &)>>> calls;
  calls[DT_INT16][DT_COMPLEX64] = FFTBaseCompute<int16_t, float>;
  calls[DT_INT32][DT_COMPLEX64] = FFTBaseCompute<int32_t, float>;
  calls[DT_INT64][DT_COMPLEX64] = FFTBaseCompute<int64_t, float>;
  calls[DT_BFLOAT16][DT_COMPLEX64] = FFTBaseCompute<bfloat16, float>;
  calls[DT_FLOAT16][DT_COMPLEX64] = FFTBaseCompute<Eigen::half, float>;
  calls[DT_FLOAT][DT_COMPLEX64] = FFTBaseCompute<float, float>;
  calls[DT_DOUBLE][DT_COMPLEX128] = FFTBaseCompute<double, double>;
  calls[DT_COMPLEX64][DT_COMPLEX64] = FFTBaseComputeC2C<complex64, float>;
  calls[DT_COMPLEX128][DT_COMPLEX128] = FFTBaseComputeC2C<complex128, double>;
  calls[DT_INT16][DT_FLOAT] = FFTBaseCompute<int16_t, float>;
  calls[DT_INT32][DT_FLOAT] = FFTBaseCompute<int32_t, float>;
  calls[DT_INT64][DT_FLOAT] = FFTBaseCompute<int64_t, float>;
  calls[DT_BFLOAT16][DT_FLOAT] = FFTBaseCompute<bfloat16, float>;
  calls[DT_FLOAT16][DT_FLOAT] = FFTBaseCompute<Eigen::half, float>;
  calls[DT_FLOAT][DT_FLOAT] = FFTBaseCompute<float, float>;
  calls[DT_DOUBLE][DT_DOUBLE] = FFTBaseCompute<double, double>;
  calls[DT_COMPLEX64][DT_FLOAT] = FFTBaseComputeC2R<complex64, float>;
  calls[DT_COMPLEX128][DT_DOUBLE] = FFTBaseComputeC2R<complex128, double>;
  return calls[input_type_][output_type_](ctx);
}

REGISTER_MS_CPU_KERNEL(kFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kRFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIRFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kHFFT, FFTBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIHFFT, FFTBaseCpuKernel);
}  // namespace aicpu
