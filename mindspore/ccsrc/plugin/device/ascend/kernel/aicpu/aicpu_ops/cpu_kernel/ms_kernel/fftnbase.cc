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
#include "common/kernel_errcode.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_kernel/utils/fft_helper.h"
#include "mindspore/core/mindapi/base/types.h"

namespace {
const char *const kInputSName = "s";
const char *const kInputDimName = "dim";
const char *const kInputNormName = "norm";
const char *kFFT2 = "FFT2";
const char *kIFFT2 = "IFFT2";
const char *kFFTN = "FFTN";
const char *kIFFTN = "IFFTN";
const char *kRFFT2 = "RFFT2";
const char *kIRFFT2 = "IRFFT2";
const char *kRFFTN = "RFFTN";
const char *kIRFFTN = "IRFFTN";
const char *kHFFT2 = "HFFT2";
const char *kIHFFT2 = "IHFFT2";
const char *kHFFTN = "HFFTN";
const char *kIHFFTN = "IHFFTN";
constexpr int kOnesideDivisor = 2;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

}  // namespace

namespace aicpu {

uint32_t FFTNBaseCpuKernel::ParseKernelParam(CpuKernelContext &ctx) {
  auto output_tensor = ctx.Output(0);
  output_type_ = output_tensor->GetDataType();
  auto input_tensor = ctx.Input(0);
  input_type_ = input_tensor->GetDataType();
  return kAicpuKernelStateSucess;
}

void FFTNBaseCpuKernel::FFTNGetAttr() {
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
  if (s_is_none_ && (op_name_ == kHFFT2 || op_name_ == kHFFTN || op_name_ == kIRFFT2 || op_name_ == kIRFFTN)) {
    s_.back() = (s_.back() - 1) * kOnesideDivisor;
  }
}

void FFTNBaseCpuKernel::FFTNGetInputs(CpuKernelContext &ctx) {
  op_name_ = GetOpName(ctx);
  tensor_shape_ = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  x_rank_ = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  if ((ctx.Input(kFftSIndex) == nullptr) || (CpuKernelUtils::GetTensorName(ctx.Input(kFftSIndex)) != kInputSName)) {
    s_is_none_ = true;
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
    norm_ = NormMode::BACKWARD;
  } else {
    norm_ = static_cast<NormMode>(reinterpret_cast<int64_t *>(ctx.Input(norm_index_)->GetData())[0]);
  }

  FFTNGetAttr();

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
  norm_weight_ = GetNormalized(fft_nums_, norm_, forward_);
}

template <typename T_in, typename T_out>
bool FFTNBaseComputeC2C(CpuKernelContext &ctx, bool forward, double norm_weight, int64_t calculate_element_nums,
                        std::vector<int64_t> tensor_shape, std::vector<int64_t> calculate_shape,
                        std::vector<int64_t> dim) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<std::complex<T_out> *>(ctx.Output(kIndex0)->GetData());

  T_out fct = static_cast<T_out>(norm_weight);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  std::complex<T_out> *calculate_input =
    static_cast<std::complex<T_out> *>(malloc(sizeof(std::complex<T_out>) * calculate_element_nums));
  auto ret = memset_s(calculate_input, sizeof(std::complex<T_out>) * calculate_element_nums, 0,
                      sizeof(std::complex<T_out>) * calculate_element_nums);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape, calculate_shape);

  // Run FFT according to parameters
  PocketFFTC2C<T_out>(calculate_input, output_ptr, forward, fct, calculate_shape, dim);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
}

template <typename T_in, typename T_out>
bool FFTNBaseComputeC2R(CpuKernelContext &ctx, bool forward, double norm_weight, int64_t calculate_element_nums_,
                        std::vector<int64_t> tensor_shape_, std::vector<int64_t> calculate_shape_,
                        std::vector<int64_t> dim_) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  auto op_name_ = GetOpName(ctx);

  T_out fct = static_cast<T_out>(norm_weight);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  std::complex<T_out> *calculate_input =
    static_cast<std::complex<T_out> *>(malloc(sizeof(std::complex<T_out>) * calculate_element_nums_));
  auto ret = memset_s(calculate_input, sizeof(std::complex<T_out>) * calculate_element_nums_, 0,
                      sizeof(std::complex<T_out>) * calculate_element_nums_);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  ShapeCopy<T_in, std::complex<T_out>>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  if (op_name_ == kHFFT2 || op_name_ == kHFFTN) {
    std::transform(calculate_input, calculate_input + calculate_element_nums_, calculate_input,
                   [](std::complex<T_out> x) { return std::conj(x); });
    forward = !forward;
  }
  // Run FFT according to parameters
  PocketFFTC2R<T_out>(calculate_input, output_ptr, forward, fct, calculate_shape_, dim_);

  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
}

template <typename T_in, typename T_out>
bool FFTNBaseComputeR2C(CpuKernelContext &ctx, bool forward, double norm_weight, int64_t calculate_element_nums_,
                        std::vector<int64_t> tensor_shape_, std::vector<int64_t> calculate_shape_,
                        std::vector<int64_t> dim_) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<std::complex<T_out> *>(ctx.Output(kIndex0)->GetData());
  auto op_name_ = GetOpName(ctx);

  T_out fct = static_cast<T_out>(norm_weight);

  // Allocate temporary memory of the required type and size and copy the input into this space.
  T_out *calculate_input = static_cast<T_out *>(malloc(sizeof(T_out) * calculate_element_nums_));
  auto ret =
    memset_s(calculate_input, sizeof(T_out) * calculate_element_nums_, 0, sizeof(T_out) * calculate_element_nums_);
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "memset_s failed.");

  ShapeCopy<T_in, T_out>(input_ptr, calculate_input, tensor_shape_, calculate_shape_);

  // Run FFT according to parameters
  PocketFFTR2C<T_out>(calculate_input, output_ptr, forward, fct, calculate_shape_, dim_);

  if (op_name_ == kIHFFT2 || op_name_ == kIHFFTN) {
    std::transform(output_ptr, output_ptr + calculate_element_nums_, output_ptr,
                   [](std::complex<T_out> x) { return std::conj(x); });
  }
  // Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;
  return KERNEL_STATUS_OK;
}

template <typename T_in, typename T_out>
uint32_t FFTNBaseCompute(CpuKernelContext &ctx, bool forward, double norm_weight, int64_t calculate_element_nums_,
                         std::vector<int64_t> tensor_shape_, std::vector<int64_t> calculate_shape_,
                         std::vector<int64_t> dim_) {
  auto op_name_ = GetOpName(ctx);
  if (op_name_ == kIHFFT2 || op_name_ == kIHFFTN) {
    forward = !forward;
    FFTNBaseComputeR2C<T_in, T_out>(ctx, forward, norm_weight, calculate_element_nums_, tensor_shape_, calculate_shape_,
                                    dim_);
  }
  if (op_name_ == kRFFT2 || op_name_ == kRFFTN) {
    FFTNBaseComputeR2C<T_in, T_out>(ctx, forward, norm_weight, calculate_element_nums_, tensor_shape_, calculate_shape_,
                                    dim_);
  }
  if (op_name_ == kHFFT2 || op_name_ == kHFFTN || op_name_ == kIRFFT2 || op_name_ == kIRFFTN) {
    FFTNBaseComputeC2R<T_in, T_out>(ctx, forward, norm_weight, calculate_element_nums_, tensor_shape_, calculate_shape_,
                                    dim_);
  }
  if (op_name_ == kFFT2 || op_name_ == kIFFT2 || op_name_ == kFFTN || op_name_ == kIFFTN) {
    FFTNBaseComputeC2C<T_in, T_out>(ctx, forward, norm_weight, calculate_element_nums_, tensor_shape_, calculate_shape_,
                                    dim_);
  }
  return KERNEL_STATUS_OK;
};

uint32_t FFTNBaseCpuKernel::Compute(CpuKernelContext &ctx) {
  ParseKernelParam(ctx);
  FFTNGetInputs(ctx);
  std::map<int, std::map<int, std::function<uint32_t(CpuKernelContext &, bool, double, int64_t, std::vector<int64_t>,
                                                     std::vector<int64_t>, std::vector<int64_t>)>>>
    calls;
  calls[DT_INT16][DT_COMPLEX64] = FFTNBaseCompute<int16_t, float>;
  calls[DT_INT32][DT_COMPLEX64] = FFTNBaseCompute<int32_t, float>;
  calls[DT_INT64][DT_COMPLEX64] = FFTNBaseCompute<int64_t, float>;
  calls[DT_FLOAT16][DT_COMPLEX64] = FFTNBaseCompute<Eigen::half, float>;
  calls[DT_FLOAT][DT_COMPLEX64] = FFTNBaseCompute<float, float>;
  calls[DT_DOUBLE][DT_COMPLEX128] = FFTNBaseCompute<double, double>;
  calls[DT_COMPLEX64][DT_COMPLEX64] = FFTNBaseComputeC2C<complex64, float>;
  calls[DT_COMPLEX128][DT_COMPLEX128] = FFTNBaseComputeC2C<complex128, double>;
  calls[DT_INT16][DT_FLOAT] = FFTNBaseCompute<int16_t, float>;
  calls[DT_INT32][DT_FLOAT] = FFTNBaseCompute<int32_t, float>;
  calls[DT_INT64][DT_FLOAT] = FFTNBaseCompute<int64_t, float>;
  calls[DT_FLOAT16][DT_FLOAT] = FFTNBaseCompute<Eigen::half, float>;
  calls[DT_FLOAT][DT_FLOAT] = FFTNBaseCompute<float, float>;
  calls[DT_DOUBLE][DT_DOUBLE] = FFTNBaseCompute<double, double>;
  calls[DT_COMPLEX64][DT_FLOAT] = FFTNBaseComputeC2R<complex64, float>;
  calls[DT_COMPLEX128][DT_DOUBLE] = FFTNBaseComputeC2R<complex128, double>;
  return calls[input_type_][output_type_](ctx, forward_, norm_weight_, calculate_element_nums_, tensor_shape_,
                                          calculate_shape_, dim_);
}

REGISTER_MS_CPU_KERNEL(kFFT2, FFTNBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFT2, FFTNBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kFFTN, FFTNBaseCpuKernel);
REGISTER_MS_CPU_KERNEL(kIFFTN, FFTNBaseCpuKernel);
}  // namespace aicpu
