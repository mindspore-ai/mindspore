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

#include "cpu_kernel/ms_kernel/dctn.h"
#include "common/kernel_errcode.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_kernel/utils/fft_helper.h"
#include "mindspore/core/mindapi/base/types.h"
#include "base/bfloat16.h"

namespace {
const char *const kInputSName = "s";
const char *const kInputDimName = "axes";
const char *const kInputNormName = "norm";
const char *kDCTN = "DCTN";
const char *kIDCTN = "IDCTN";

constexpr double kDCTFactor = 2.0;
constexpr int64_t kNormFactor = 2;
constexpr int64_t kDCTType = 2;
constexpr int64_t kIDCTType = 3;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

}  // namespace

namespace aicpu {

uint32_t DCTNCpuKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto output_tensor = ctx.Output(0);
  output_type_ = output_tensor->GetDataType();
  auto input_tensor = ctx.Input(0);
  input_type_ = input_tensor->GetDataType();
  return kAicpuKernelStateSucess;
}

void DCTNCpuKernel::DCTNGetAttr() {
  if (s_is_none_ && !dim_is_none_) {
    for (size_t i = 0; i < dim_.size(); i++) {
      (void)s_.emplace_back(tensor_shape_[dim_[i]]);
    }
  }
  if (dim_is_none_ && !s_is_none_) {
    for (size_t i = 0; i < s_.size(); i++) {
      (void)dim_.emplace_back(x_rank_ - s_.size() + i);
    }
  }
  if (dim_is_none_ && s_is_none_) {
    for (int64_t i = 0; i < x_rank_; i++) {
      (void)dim_.emplace_back(i);
      (void)s_.emplace_back(tensor_shape_[i]);
    }
  }
}

void DCTNCpuKernel::DCTNGetInputs(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx);
  tensor_shape_ = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  x_rank_ = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  dct_type_ = (op_name_ == kDCTN) ? kDCTType : kIDCTType;

  if ((ctx.Input(kDCTSIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kDCTSIndex)) != kInputSName)) {
    s_is_none_ = true;
    dim_index_--;
    norm_index_--;
  } else {
    auto s_tensor = reinterpret_cast<int64_t *>(ctx.Input(kDCTSIndex)->GetData());
    auto s_tensor_size = ctx.Input(kDCTSIndex)->NumElements();
    for (int64_t i = 0; i < s_tensor_size; i++) {
      (void)s_.emplace_back(s_tensor[i]);
    }
  }
  if ((ctx.Input(dim_index_) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(dim_index_)) != kInputDimName)) {
    dim_is_none_ = true;
    dim_index_--;
  } else {
    auto dim_tensor = reinterpret_cast<int64_t *>(ctx.Input(dim_index_)->GetData());
    auto dim_tensor_size = ctx.Input(dim_index_)->NumElements();
    for (int64_t i = 0; i < dim_tensor_size; i++) {
      int64_t tmp_pos = dim_tensor[i] < 0 ? x_rank_ + dim_tensor[i] : dim_tensor[i];
      (void)dim_.emplace_back(tmp_pos);
    }
  }

  bool norm_is_none =
    (ctx.Input(norm_index_) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(norm_index_)) != kInputNormName);
  if (norm_is_none) {
    norm_ = NormMode::ORTHO;
  } else {
    norm_ = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index_)->GetData())[0]);
  }
  is_ortho_ = (norm_ == NormMode::ORTHO);

  DCTNGetAttr();

  forward_ = IsForwardOp(op_name_);
  input_element_nums_ = ctx.Input(kIndex0)->NumElements();

  calculate_shape_.clear();
  calculate_element_nums_ = GetCalculateElementNum(tensor_shape_, dim_, s_, input_element_nums_);
  for (size_t i = 0; i < tensor_shape_.size(); i++) {
    (void)calculate_shape_.emplace_back(tensor_shape_[i]);
  }
  for (size_t i = 0; i < dim_.size(); i++) {
    calculate_shape_[dim_[i]] = s_[i];
  }

  fft_nums_ = std::accumulate(s_.begin(), s_.end(), 1, std::multiplies<int64_t>());
  auto cmpt_nums = fft_nums_ * pow(kNormFactor, static_cast<int64_t>(s_.size()));
  norm_weight_ = GetNormalized(cmpt_nums, norm_, forward_);
}

template <typename T_in, typename T_out>
uint32_t DCTNCompute(CpuKernelContext &ctx, int64_t dct_type, double norm_weight, int64_t calculate_element_nums,
                     std::vector<int64_t> tensor_shape, std::vector<int64_t> calculate_shape, std::vector<int64_t> dim,
                     bool is_ortho) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  auto op_name_ = GetOpName(ctx);

  T_out fct = static_cast<T_out>(norm_weight);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums));
  auto ret =
    memset_s(calculate_input, sizeof(T_out) * calculate_element_nums, 0, sizeof(T_out) * calculate_element_nums);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  ShapeCopy<T_in, T_out>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // Run FFT according to parameters
  PocketFFTDCT<T_out>(calculate_input, output_ptr, dct_type, fct, calculate_shape, dim, is_ortho);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
};

template <typename T_in, typename T_out>
uint32_t DCTNComputeComplex(CpuKernelContext &ctx, int64_t dct_type, double norm_weight, int64_t calculate_element_nums,
                            std::vector<int64_t> tensor_shape, std::vector<int64_t> calculate_shape,
                            std::vector<int64_t> dim, bool is_ortho) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_in *>(ctx.Output(kIndex0)->GetData());
  auto op_name_ = GetOpName(ctx);

  T_out fct = static_cast<T_out>(norm_weight);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums));
  T_out *calculate_input_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums));
  auto ret_real =
    memset_s(calculate_input_real, sizeof(T_out) * calculate_element_nums, 0, sizeof(T_out) * calculate_element_nums);
  auto ret_imag =
    memset_s(calculate_input_imag, sizeof(T_out) * calculate_element_nums, 0, sizeof(T_out) * calculate_element_nums);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret_real == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");
  CUST_KERNEL_CHECK_FALSE(ctx, (ret_imag == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  ShapeCopy<T_in, T_out>(input_ptr, calculate_input_real, tensor_shape, calculate_shape);
  ShapeCopy<T_in, T_out>(input_ptr, calculate_input_imag, tensor_shape, calculate_shape, false);

  // Run FFT according to parameters
  T_out *output_real = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums));
  T_out *output_imag = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums));
  PocketFFTDCT<T_out>(calculate_input_real, output_real, dct_type, fct, calculate_shape, dim, is_ortho);
  PocketFFTDCT<T_out>(calculate_input_imag, output_imag, dct_type, fct, calculate_shape, dim, is_ortho);

  for (int64_t i = 0; i < calculate_element_nums; ++i) {
    std::complex<T_out> temp_val{*(output_real + i), *(output_imag + i)};
    *(output_ptr + i) = temp_val;
  }

  // Release temporary memory
  free(calculate_input_real);
  free(calculate_input_imag);
  calculate_input_real = nullptr;
  calculate_input_imag = nullptr;
  return KERNEL_STATUS_OK;
};

uint32_t DCTNCpuKernel::Compute(CpuKernelContext &ctx) {
  ParseKernelParam(ctx);
  DCTNGetInputs(ctx);
  std::map<int, std::map<int, std::function<uint32_t(CpuKernelContext &, int64_t, double, int64_t, std::vector<int64_t>,
                                                     std::vector<int64_t>, std::vector<int64_t>, bool)>>>
    calls;
  calls[DT_INT16][DT_FLOAT] = DCTNCompute<int16_t, float>;
  calls[DT_INT32][DT_FLOAT] = DCTNCompute<int32_t, float>;
  calls[DT_INT64][DT_FLOAT] = DCTNCompute<int64_t, float>;
  calls[DT_BFLOAT16][DT_FLOAT] = DCTNCompute<bfloat16, float>;
  calls[DT_FLOAT16][DT_FLOAT] = DCTNCompute<Eigen::half, float>;
  calls[DT_FLOAT][DT_FLOAT] = DCTNCompute<float, float>;
  calls[DT_DOUBLE][DT_DOUBLE] = DCTNCompute<double, double>;
  calls[DT_COMPLEX64][DT_COMPLEX64] = DCTNComputeComplex<complex64, float>;
  calls[DT_COMPLEX128][DT_COMPLEX128] = DCTNComputeComplex<complex128, double>;
  return calls[input_type_][output_type_](ctx, dct_type_, norm_weight_, calculate_element_nums_, tensor_shape_,
                                          calculate_shape_, dim_, is_ortho_);
}

REGISTER_MS_CPU_KERNEL(kDCTN, DCTNCpuKernel);
REGISTER_MS_CPU_KERNEL(kIDCTN, DCTNCpuKernel);
}  // namespace aicpu
