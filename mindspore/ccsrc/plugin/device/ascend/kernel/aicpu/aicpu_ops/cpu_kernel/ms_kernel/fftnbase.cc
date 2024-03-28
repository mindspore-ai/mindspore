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

#include "cpu_kernel/ms_kernel/fftnbase.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_kernel/utils/fft_helper.h"
#include "mindspore/core/mindapi/base/types.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
const char *const kInputSName = "s";
const char *const kInputDimName = "dim";
const char *const kInputNormName = "norm";
const char *kFFT2 = "FFT2";
const char *kFFTN = "FFTN";
const char *kIFFT2 = "IFFT2";
const char *kIFFTN = "IFFTN";

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

#define FFTNBASE_COMPUTE_CASE(DTYPE, INTYPE, MIDTYPE, OUTTYPE, CTX)   \
  case (DTYPE): {                                                     \
    uint32_t result = FFTNBaseCompute<INTYPE, MIDTYPE, OUTTYPE>(CTX); \
    if (result != KERNEL_STATUS_OK) {                                 \
      CUST_KERNEL_LOG_ERROR(ctx, "FFTNBase kernel compute failed.");  \
      return result;                                                  \
    }                                                                 \
    break;                                                            \
  }

}  // namespace

namespace aicpu {
uint32_t FFTNBaseCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx.GetOpType());
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           op_name_.c_str());
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    FFTNBASE_COMPUTE_CASE(DT_INT16, int16_t, float, complex64, ctx)
    FFTNBASE_COMPUTE_CASE(DT_INT32, int32_t, float, complex64, ctx)
    FFTNBASE_COMPUTE_CASE(DT_INT64, int64_t, float, complex64, ctx)
    FFTNBASE_COMPUTE_CASE(DT_FLOAT16, Eigen::half, float, complex64, ctx)
    FFTNBASE_COMPUTE_CASE(DT_FLOAT, float, float, complex64, ctx)
    FFTNBASE_COMPUTE_CASE(DT_DOUBLE, double, double, complex128, ctx)
    FFTNBASE_COMPUTE_CASE(DT_COMPLEX64, complex64, complex64, complex64, ctx)
    FFTNBASE_COMPUTE_CASE(DT_COMPLEX128, complex128, complex128, complex128, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "FFTNBase kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

#define FFTN_INPUT_DIM_CASE(T1, T2)                                                                     \
  if (x_rank == 1) {                                                                                    \
    EigenFFTNBase<T1, T2, 1>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else if (x_rank == 2) {                                                                             \
    EigenFFTNBase<T1, T2, 2>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else if (x_rank == 3) {                                                                             \
    EigenFFTNBase<T1, T2, 3>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else if (x_rank == 4) {                                                                             \
    EigenFFTNBase<T1, T2, 4>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else if (x_rank == 5) {                                                                             \
    EigenFFTNBase<T1, T2, 5>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else if (x_rank == 6) {                                                                             \
    EigenFFTNBase<T1, T2, 6>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else if (x_rank == 7) {                                                                             \
    EigenFFTNBase<T1, T2, 7>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  } else {                                                                                              \
    EigenFFTNBase<T1, T2, 8>(calculate_input, output_ptr, forward, norm_weight, calculate_shape, dim_); \
  }

template <typename T_in, typename T_out, int x_rank>
bool EigenFFTNBase(T_in *input_ptr, T_out *output_ptr, bool forward, double norm_weight,
                   std::vector<int64_t> tensor_shape, std::vector<int64_t> dim) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape_array;
  for (size_t i = 0; i < x_rank; ++i) {
    tensor_shape_array[i] = tensor_shape[i];
  }

  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0], tensor_shape_array);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  std::vector<int32_t> eigem_dim;
  for (size_t i = 0; i < dim.size(); i++) {
    (void)eigem_dim.emplace_back(static_cast<int32_t>(dim[i]));
  }

  if (forward) {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(eigem_dim);
  } else {
    out = in.template fft<Eigen::BothParts, Eigen::FFT_REVERSE>(eigem_dim);
  }

  T_out *out_ptr = out.data();
  for (int i = 0; i < out.size(); i++) {
    T_out temp_value = *(out_ptr + i);
    temp_value *= norm_weight;
    *(output_ptr + i) = temp_value;
  }
  return true;
}

void FFTNBaseCpuKernel::FFTNGetInputs(CpuKernelContext &ctx) {
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();
  if ((ctx.Input(kFftSIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kFftSIndex)) != kInputSName)) {
    s_is_null_ = true;
    dim_index_--;
    norm_index_--;
  } else {
    auto s_tensor = reinterpret_cast<int64_t *>(ctx.Input(kFftSIndex)->GetData());
    auto s_tensor_size = ctx.Input(kFftSIndex)->NumElements();
    for (int64_t i = 0; i < s_tensor_size; i++) {
      (void)s_.emplace_back(s_tensor[i]);
    }
  }
  if ((ctx.Input(dim_index_) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(dim_index_)) != kInputDimName)) {
    dim_is_null_ = true;
    dim_index_--;
  } else {
    auto dim_tensor = reinterpret_cast<int64_t *>(ctx.Input(dim_index_)->GetData());
    auto dim_tensor_size = ctx.Input(dim_index_)->NumElements();
    for (int64_t i = 0; i < dim_tensor_size; i++) {
      int64_t tmp_pos = dim_tensor[i] < 0 ? x_rank + dim_tensor[i] : dim_tensor[i];
      (void)dim_.emplace_back(tmp_pos);
    }
  }

  if (s_is_null_ && !dim_is_null_) {
    for (size_t i = 0; i < dim_.size(); i++) {
      (void)s_.emplace_back(tensor_shape[dim_[i]]);
    }
  }
  if (dim_is_null_ && !s_is_null_) {
    for (size_t i = 0; i < s_.size(); i++) {
      (void)dim_.emplace_back(x_rank - s_.size() + i);
    }
  }
  if (dim_is_null_ && s_is_null_) {
    for (int64_t i = 0; i < x_rank; i++) {
      (void)dim_.emplace_back(i);
      (void)s_.emplace_back(tensor_shape[i]);
    }
  }
}

template <typename T_in, typename T_mid, typename T_out>
uint32_t FFTNBaseCpuKernel::FFTNBaseCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();
  bool forward = IsForwardOp(op_name_);
  int64_t input_element_nums{ctx.Input(kIndex0)->NumElements()};

  FFTNGetInputs(ctx);
  // step1：Get or set attribute.
  NormMode norm;
  bool norm_is_none =
    (ctx.Input(norm_index_) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(norm_index_)) != kInputNormName);
  if (norm_is_none) {
    norm = NormMode::BACKWARD;
  } else {
    norm = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index_)->GetData())[0]);
  }
  // step2：Calculate the required memory based on n and dim.
  int64_t calculate_element = GetCalculateElementNum(tensor_shape, dim_, s_, input_element_nums);
  std::vector<int64_t> calculate_shape;
  for (size_t i = 0; i < tensor_shape.size(); i++) {
    (void)calculate_shape.emplace_back(tensor_shape[i]);
  }
  for (size_t i = 0; i < dim_.size(); i++) {
    calculate_shape[dim_[i]] = s_[i];
  }

  int64_t fft_nums = std::accumulate(s_.begin(), s_.end(), 1, std::multiplies<int64_t>());
  double norm_weight = GetNormalized(fft_nums, norm, forward);

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element));
  auto ret = memset_s(calculate_input, sizeof(T_mid) * calculate_element, 0, sizeof(T_mid) * calculate_element);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  ShapeCopy<T_in, T_mid>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // step4：Run FFTN according to parameters
  FFTN_INPUT_DIM_CASE(T_mid, T_out);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kFFTN, FFTNBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kFFT2, FFTNBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFTN, FFTNBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFT2, FFTNBaseCpuKernel);
}  // namespace aicpu
