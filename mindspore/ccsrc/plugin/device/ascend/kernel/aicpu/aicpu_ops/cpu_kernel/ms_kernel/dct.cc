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

#include "cpu_kernel/ms_kernel/dct.h"
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
const char *kDCT = "DCT";
const char *kIDCT = "IDCT";

constexpr double kDCTFactor = 2.0;
constexpr int64_t kNormFactor = 2;
constexpr int64_t kDCTType = 2;
constexpr int64_t kIDCTType = 3;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

uint32_t DCTCpuKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto output_tensor = ctx.Output(0);
  output_type_ = output_tensor->GetDataType();
  auto input_tensor = ctx.Input(0);
  input_type_ = input_tensor->GetDataType();
  return kAicpuKernelStateSucess;
}

template <typename T_in, typename T_out>
uint32_t DCTCpuKernel::DCTCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  std::size_t dim_index = kDCTDimIndex;
  std::size_t norm_index = kDCTNormIndex;
  auto op_name = GetOpName(ctx);

  // step1：optional input will be null, we need to adapt the corresponding input index
  int64_t dct_type = (op_name == kDCT) ? kDCTType : kIDCTType;
  int64_t n;
  bool n_is_none =
    (ctx.Input(kDCTNIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kDCTNIndex)) != kInputNName);
  if (n_is_none) {
    dim_index--;
    norm_index--;
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kDCTNIndex)->GetData())[0];
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
    norm = NormMode::ORTHO;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index)->GetData())[0]);
  }
  bool is_ortho = (norm == NormMode::ORTHO);

  bool forward = IsForwardOp(op_name);
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;
  auto cmpt_num = kNormFactor * n;
  double norm_weight = GetNormalized(cmpt_num, norm, forward);
  T_out fct = static_cast<T_out>(norm_weight);

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element));
  auto ret = memset_s(calculate_input, sizeof(T_out) * calculate_element, 0, sizeof(T_out) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // step4：Run FFT according to parameters
  std::vector<int64_t> dim_vec(1, dim);
  PocketFFTDCT<T_out>(calculate_input, output_ptr, dct_type, fct, calculate_shape, dim_vec, is_ortho);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
};

template <typename T_in, typename T_out>
uint32_t DCTCpuKernel::DCTComputeComplex(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_in *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  std::size_t dim_index = kDCTDimIndex;
  std::size_t norm_index = kDCTNormIndex;
  auto op_name = GetOpName(ctx);

  // step1：optional input will be null, we need to adapt the corresponding input index
  int64_t dct_type = (op_name == kDCT) ? kDCTType : kIDCTType;
  int64_t n;
  bool n_is_none =
    (ctx.Input(kDCTNIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kDCTNIndex)) != kInputNName);
  if (n_is_none) {
    dim_index--;
    norm_index--;
  } else {
    n = reinterpret_cast<int64_t *>(ctx.Input(kDCTNIndex)->GetData())[0];
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
    norm = NormMode::ORTHO;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index)->GetData())[0]);
  }
  bool is_ortho = (norm == NormMode::ORTHO);

  bool forward = IsForwardOp(op_name);
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;
  auto cmpt_num = kNormFactor * n;
  double norm_weight = GetNormalized(cmpt_num, norm, forward);
  T_out fct = static_cast<T_out>(norm_weight);

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element));
  T_out *calculate_input_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element));
  auto ret_real =
    memset_s(calculate_input_real, sizeof(T_out) * calculate_element, 0, sizeof(T_out) * calculate_element);
  auto ret_imag =
    memset_s(calculate_input_imag, sizeof(T_out) * calculate_element, 0, sizeof(T_out) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret_real == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  CUST_KERNEL_CHECK_FALSE(ctx, (ret_imag == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input_real, tensor_shape, calculate_shape);
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input_imag, tensor_shape, calculate_shape, false);

  // step4：Run FFT according to parameters
  T_out *output_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element));
  T_out *output_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element));
  std::vector<int64_t> dim_vec(1, dim);
  PocketFFTDCT<T_out>(calculate_input_real, output_real, dct_type, fct, calculate_shape, dim_vec, is_ortho);
  PocketFFTDCT<T_out>(calculate_input_imag, output_imag, dct_type, fct, calculate_shape, dim_vec, is_ortho);

  for (int64_t i = 0; i < calculate_element; ++i) {
    std::complex<T_out> temp_val{*(output_real + i), *(output_imag + i)};
    *(output_ptr + i) = temp_val;
  }
  // step5: Release temporary memory
  free(calculate_input_real);
  free(calculate_input_imag);
  calculate_input_real = nullptr;
  calculate_input_imag = nullptr;
  return KERNEL_STATUS_OK;
};

uint32_t DCTCpuKernel::Compute(CpuKernelContext &ctx) {
  ParseKernelParam(ctx);
  std::map<int, std::map<int, std::function<uint32_t(CpuKernelContext &)>>> calls;
  calls[DT_INT16][DT_FLOAT] = DCTCompute<int16_t, float>;
  calls[DT_INT32][DT_FLOAT] = DCTCompute<int32_t, float>;
  calls[DT_INT64][DT_FLOAT] = DCTCompute<int64_t, float>;
  calls[DT_BFLOAT16][DT_FLOAT] = DCTCompute<bfloat16, float>;
  calls[DT_FLOAT16][DT_FLOAT] = DCTCompute<Eigen::half, float>;
  calls[DT_FLOAT][DT_FLOAT] = DCTCompute<float, float>;
  calls[DT_DOUBLE][DT_DOUBLE] = DCTCompute<double, double>;
  calls[DT_COMPLEX64][DT_COMPLEX64] = DCTComputeComplex<complex64, float>;
  calls[DT_COMPLEX128][DT_COMPLEX128] = DCTComputeComplex<complex128, double>;
  return calls[input_type_][output_type_](ctx);
}

REGISTER_MS_CPU_KERNEL(kDCT, DCTCpuKernel);
REGISTER_MS_CPU_KERNEL(kIDCT, DCTCpuKernel);
}  // namespace aicpu
