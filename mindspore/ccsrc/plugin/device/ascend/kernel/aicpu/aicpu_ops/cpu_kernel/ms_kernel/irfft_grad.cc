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

#include "cpu_kernel/ms_kernel/irfft_grad.h"
#include <securec.h>
#include "unsupported/Eigen/CXX11/Tensor"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "mindspore/core/mindapi/base/types.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 2;
const uint32_t kIndex0 = 0;
const uint32_t kInput2Index0 = 1;
const uint32_t kNIndex = 2;
const uint32_t kDimIndex = 3;
const uint32_t kNormIndex = 4;
const int kRealFFTSideNum = 2;
constexpr double kDoubleFactor = 2.0;

const char *kIRFFTGrad = "IRFFTGrad";
const std::string op_prefix = "Cust";
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using NormMode = mindspore::NormMode;

#define IRFFTGRAD_COMPUTE_CASE(DTYPE, INTYPE, MIDTYPE, OUTTYPE, CTX)   \
  case (DTYPE): {                                                      \
    uint32_t result = IRFFTGradCompute<INTYPE, MIDTYPE, OUTTYPE>(CTX); \
    if (result != KERNEL_STATUS_OK) {                                  \
      CUST_KERNEL_LOG_ERROR(ctx, "IRFFTGrad kernel compute failed.");  \
      return result;                                                   \
    }                                                                  \
    break;                                                             \
  }

}  // namespace

namespace aicpu {
uint32_t IRFFTGradCpuKernel::Compute(CpuKernelContext &ctx) {
  op_name = ctx.GetOpType();
  if (op_name.find(op_prefix) == 0) {
    op_name.erase(op_name.begin(), op_name.begin() + op_prefix.size());
  }
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.",
                           op_name.c_str());
  auto x_type = ctx.Input(kIndex0)->GetDataType();
  switch (x_type) {
    //    IRFFTGRAD_COMPUTE_CASE(DT_INT16, int16_t, float, complex64, ctx)
    //    IRFFTGRAD_COMPUTE_CASE(DT_INT32, int32_t, float, complex64, ctx)
    //    IRFFTGRAD_COMPUTE_CASE(DT_INT64, int64_t, float, complex64, ctx)
    //    IRFFTGRAD_COMPUTE_CASE(DT_FLOAT16, Eigen::half, float, complex64, ctx)
    IRFFTGRAD_COMPUTE_CASE(DT_FLOAT, float, float, complex64, ctx)
    IRFFTGRAD_COMPUTE_CASE(DT_DOUBLE, double, double, complex128, ctx)
      //    IRFFTGRAD_COMPUTE_CASE(DT_COMPLEX64, complex64, complex64, complex64, ctx)
      //    IRFFTGRAD_COMPUTE_CASE(DT_COMPLEX128, complex128, complex128, complex128, ctx)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "IRFFTGrad kernel data type [%s] not support.", DTypeStr(x_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

#define SWITCH_DIM_CALCULATE(T1, T2)                                                                           \
  if (x_rank == 1) {                                                                                           \
    ComputeIRFFTGrad<T1, T2, 1>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else if (x_rank == 2) {                                                                                    \
    ComputeIRFFTGrad<T1, T2, 2>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else if (x_rank == 3) {                                                                                    \
    ComputeIRFFTGrad<T1, T2, 3>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else if (x_rank == 4) {                                                                                    \
    ComputeIRFFTGrad<T1, T2, 4>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else if (x_rank == 5) {                                                                                    \
    ComputeIRFFTGrad<T1, T2, 5>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else if (x_rank == 6) {                                                                                    \
    ComputeIRFFTGrad<T1, T2, 6>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else if (x_rank == 7) {                                                                                    \
    ComputeIRFFTGrad<T1, T2, 7>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  } else {                                                                                                     \
    ComputeIRFFTGrad<T1, T2, 8>(calculate_input, output_ptr, norm, n, dim, calculate_shape, calculate_element, \
                                tensor2_shape);                                                                \
  }

double IRFFTGradGetnormalized(int64_t element_nums, NormMode norm_type_) {
  double result = 1.0;
  if (norm_type_ == NormMode::BACKWARD) {
    result = 1.0 / element_nums;
  } else if (norm_type_ == NormMode::ORTHO) {
    result = 1.0 / sqrt(static_cast<double>(element_nums));
  }
  return result;
}

template <typename T_in, typename T_out>
void IRFFTGradGenerateCalculateInput(T_in *array_in, T_out *array_out, int64_t element_nums,
                                     const std::vector<int64_t> &x_shape, const std::vector<int64_t> &calculate_shape,
                                     int64_t n, int64_t dim) {
  // compute original and new offsets for each dim
  std::vector<int64_t> offsets(x_shape.size(), 0);
  std::vector<int64_t> new_offsets(x_shape.size(), 0);
  for (size_t j = 0; j < x_shape.size(); j++) {
    offsets[j] = std::accumulate(x_shape.begin() + j + 1, x_shape.end(), 1, std::multiplies<>());
    new_offsets[j] = std::accumulate(calculate_shape.begin() + j + 1, calculate_shape.end(), 1, std::multiplies<>());
  }

  for (int64_t i = 0; i < element_nums; ++i) {
    std::vector<int64_t> index(x_shape.size(), 0);
    int64_t flat_index = i;
    // compute original coordinates
    for (size_t dim_index = 0; dim_index < offsets.size(); ++dim_index) {
      index[dim_index] = flat_index / offsets[dim_index];
      flat_index %= offsets[dim_index];
    }
    // if n > input.shape[dim] ->truncate, invalid ele should be dropped out
    if (index[dim] >= n) {
      continue;
    }
    int64_t new_flat_index = 0;
    for (size_t dim_index = 0; dim_index < new_offsets.size(); ++dim_index) {
      new_flat_index += index[dim_index] * new_offsets[dim_index];
    }
    array_out[new_flat_index] = static_cast<T_out>(array_in[i]);
  }
}

template <typename T_in, typename T_out, int x_rank>
bool ComputeIRFFTGrad(T_in *input_ptr, T_out *output_ptr, NormMode norm_type, int64_t n, int64_t dim,
                      std::vector<int64_t> x_shape, int64_t element_nums, std::vector<int64_t> out_shape) {
  Eigen::array<Eigen::DenseIndex, x_rank> tensor_shape;
  for (int i = 0; i < x_rank; ++i) {
    tensor_shape[i] = x_shape[i];
  }
  Eigen::TensorMap<Eigen::Tensor<T_in, x_rank, Eigen::RowMajor>, Eigen::RowMajor> in(&input_ptr[0], tensor_shape);
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> out;

  Eigen::array<int, 1> dims_array;
  dims_array[0] = dim;
  out = in.template fft<Eigen::BothParts, Eigen::FFT_FORWARD>(dims_array);

  // rfft slice
  Eigen::DSizes<Eigen::DenseIndex, x_rank> double_offsets;
  Eigen::array<Eigen::DenseIndex, x_rank> double_slice_sizes(in.dimensions());
  auto dims = in.dimensions();
  Eigen::DSizes<Eigen::DenseIndex, x_rank> offsets;
  Eigen::DSizes<Eigen::DenseIndex, x_rank> input_slice_sizes;
  for (auto i = 0; i < x_rank; i++) {
    offsets[i] = 0;
    double_offsets[i] = (i == dim) ? 1 : 0;
    input_slice_sizes[i] = (i == dim) ? (dims[i] / kRealFFTSideNum + 1) : dims[i];
    double_slice_sizes[i] = (i == dim) ? (dims[i] - (dims[i] / kRealFFTSideNum + 1)) : dims[i];
  }
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> slice_out = out.slice(offsets, input_slice_sizes);
  out = slice_out;
  double norm_weight = IRFFTGradGetnormalized(n, norm_type);

  // double input.slice(dim, (1, input.shape[dim] - input_slice_sizes[dim]))
  T_out factor(kDoubleFactor);
  out.slice(double_offsets, double_slice_sizes) = (out.slice(double_offsets, double_slice_sizes) * factor);

  // padding or trimmed back to input2's shape
  Eigen::array<Eigen::DenseIndex, x_rank> final_out_shape;
  for (int i = 0; i < x_rank; ++i) {
    final_out_shape[i] = out_shape[i];
  }
  Eigen::Tensor<T_out, x_rank, Eigen::RowMajor> final_out(final_out_shape);
  final_out.setZero();
  Eigen::array<Eigen::DenseIndex, x_rank> slice_sizes(out.dimensions());
  for (auto i = 0; i < x_rank; i++) {
    slice_sizes[i] = std::min(out_shape[i], static_cast<int64_t>(slice_sizes[i]));
  }
  final_out.slice(offsets, slice_sizes) = out.slice(offsets, slice_sizes);

  T_out *out_ptr = final_out.data();
  for (int i = 0; i < final_out.size(); i++) {
    T_out temp_value = *(out_ptr + i);
    temp_value *= norm_weight;
    *(output_ptr + i) = temp_value;
  }

  return true;
}

template <typename T_in, typename T_mid, typename T_out>
uint32_t IRFFTGradCpuKernel::IRFFTGradCompute(CpuKernelContext &ctx) {
  auto input_ptr = reinterpret_cast<T_in *>(ctx.Input(kIndex0)->GetData());
  auto output_ptr = reinterpret_cast<T_out *>(ctx.Output(kIndex0)->GetData());
  std::vector<int64_t> tensor_shape = ctx.Input(kIndex0)->GetTensorShape()->GetDimSizes();
  int64_t x_rank = ctx.Input(kIndex0)->GetTensorShape()->GetDims();

  std::vector<int64_t> tensor2_shape = ctx.Input(kInput2Index0)->GetTensorShape()->GetDimSizes();

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

  // step2：Calculate the required memory based on n and dim.
  int64_t input_element{ctx.Input(kIndex0)->NumElements()};
  int64_t calculate_element = input_element / tensor_shape[dim] * n;
  std::vector<int64_t> calculate_shape(tensor_shape.begin(), tensor_shape.end());
  calculate_shape[dim] = n;

  // step3：Allocate temporary memory of the required type and size and copy the input into this space.
  T_mid *calculate_input = static_cast<T_mid *>(malloc(sizeof(T_mid) * calculate_element));
  if (memset_s(calculate_input, sizeof(T_mid) * calculate_element, 0, sizeof(T_mid) * calculate_element) != EOK) {
    free(calculate_input);
    calculate_input = nullptr;
    CUST_KERNEL_LOG_ERROR(ctx, "For 'IRFFTGrad', memset_s failed. ");
    return KERNEL_STATUS_INNER_ERROR;
  }
  IRFFTGradGenerateCalculateInput<T_in, T_mid>(input_ptr, calculate_input, input_element, tensor_shape, calculate_shape,
                                               n, dim);

  // step4：Run FFT according to parameters
  SWITCH_DIM_CALCULATE(T_mid, T_out);

  // step5: Release temporary memory
  free(calculate_input);
  calculate_input = nullptr;

  return KERNEL_STATUS_OK;
};

REGISTER_MS_CPU_KERNEL(kIRFFTGrad, IRFFTGradCpuKernel);
}  // namespace aicpu