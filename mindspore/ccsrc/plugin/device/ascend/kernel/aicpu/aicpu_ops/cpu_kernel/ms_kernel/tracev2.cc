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

#include "cpu_kernel/ms_kernel/tracev2.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include <algorithm>
#include <complex>
#include "cstring"
#include "securec.h"
#include "context/inc/cpu_kernel_utils.h"
#include "cpu_kernel/utils/linalg_utils.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 1;
constexpr size_t kIndexIn = 0;
constexpr size_t kIndexOffset = 1;
constexpr size_t kIndexAxis1 = 2;
constexpr size_t kIndexAxis2 = 3;
constexpr size_t kIndexDtype = 4;
constexpr size_t kIndexOut = 0;
constexpr size_t kMaxDim = 8;
const char *kTraceV2 = "TraceV2";
}  // namespace

namespace aicpu {

template <typename T>
void TraceV2CpuKernel::RegFunc(DataType dtype) {
  calls_[dtype][DT_UINT8] = TraceV2Compute<T, uint8_t>;
  calls_[dtype][DT_UINT16] = TraceV2Compute<T, uint16_t>;
  calls_[dtype][DT_UINT32] = TraceV2Compute<T, uint32_t>;
  calls_[dtype][DT_UINT64] = TraceV2Compute<T, uint64_t>;
  calls_[dtype][DT_INT8] = TraceV2Compute<T, int8_t>;
  calls_[dtype][DT_INT16] = TraceV2Compute<T, int16_t>;
  calls_[dtype][DT_INT32] = TraceV2Compute<T, int32_t>;
  calls_[dtype][DT_INT64] = TraceV2Compute<T, int64_t>;
  calls_[dtype][DT_FLOAT16] = TraceV2Compute<T, float16>;
  calls_[dtype][DT_FLOAT] = TraceV2Compute<T, float>;
  calls_[dtype][DT_DOUBLE] = TraceV2Compute<T, double>;
  calls_[dtype][DT_COMPLEX64] = TraceV2Compute<T, complex64>;
  calls_[dtype][DT_COMPLEX128] = TraceV2Compute<T, complex128>;
  calls_[dtype][DT_BOOL] = TraceV2Compute<T, bool>;
  calls_[dtype][DT_BFLOAT16] = TraceV2Compute<T, bfloat16>;
}

uint32_t TraceV2CpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "TraceV2 check input and output number failed.");

  Tensor *input_tensor = ctx.Input(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_tensor, KERNEL_STATUS_PARAM_INVALID, "TraceV2 get input failed.")

  // check output tensor
  Tensor *output_tensor = ctx.Output(0);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output_tensor, KERNEL_STATUS_PARAM_INVALID, "TraceV2 get output failed.")

  auto input_dtype = input_tensor->GetDataType();
  auto output_dtype = output_tensor->GetDataType();
  RegFunc<uint8_t>(DT_UINT8);
  RegFunc<uint16_t>(DT_UINT16);
  RegFunc<uint32_t>(DT_UINT32);
  RegFunc<uint64_t>(DT_UINT64);
  RegFunc<int8_t>(DT_INT8);
  RegFunc<int16_t>(DT_INT16);
  RegFunc<int32_t>(DT_INT32);
  RegFunc<int64_t>(DT_INT64);
  RegFunc<float16>(DT_FLOAT16);
  RegFunc<float>(DT_FLOAT);
  RegFunc<double>(DT_DOUBLE);
  RegFunc<complex64>(DT_COMPLEX64);
  RegFunc<complex128>(DT_COMPLEX128);
  RegFunc<bool>(DT_BOOL);
  RegFunc<bfloat16>(DT_BFLOAT16);
  return calls_[input_dtype][output_dtype](ctx);
}

template <typename T>
void transpose_step(T *in_addr, T *out_addr, std::vector<int64_t> shape_in, int64_t axis1, int64_t axis2) {
  int64_t in_rank = shape_in.size();
  size_t trans_offset = kMaxDim - in_rank;
  std::vector<int64_t> shape_out;
  std::vector<int64_t> perm_vec;
  std::vector<int64_t> trans_in_shape(kMaxDim, 1);
  std::vector<int64_t> trans_out_shape(kMaxDim, 1);
  for (int64_t i = 0; i < in_rank; i++) {
    if (i != axis1 && i != axis2) {
      shape_out.emplace_back(shape_in[i]);
      perm_vec.emplace_back(i);
    }
  }
  shape_out.emplace_back(shape_in[axis1]);
  shape_out.emplace_back(shape_in[axis2]);
  perm_vec.emplace_back(axis1);
  perm_vec.emplace_back(axis2);

  for (int64_t i = 0; i < in_rank; i++) {
    trans_in_shape[i + trans_offset] = shape_in[i];
    trans_out_shape[i + trans_offset] = shape_out[i];
  }
  using Eigen_Tensor = Eigen::TensorMap<Eigen::Tensor<T, kMaxDim, Eigen::RowMajor>, Eigen::Aligned>;

  Eigen_Tensor trans_input(in_addr, trans_in_shape.at(0), trans_in_shape.at(1), trans_in_shape.at(2),
                           trans_in_shape.at(3), trans_in_shape.at(4), trans_in_shape.at(5), trans_in_shape.at(6),
                           trans_in_shape.at(7));
  Eigen_Tensor trans_output(out_addr, trans_out_shape.at(0), trans_out_shape.at(1), trans_out_shape.at(2),
                            trans_out_shape.at(3), trans_out_shape.at(4), trans_out_shape.at(5), trans_out_shape.at(6),
                            trans_out_shape.at(7));
  Eigen::array<Eigen::DenseIndex, kMaxDim> perm_compute;
  for (size_t j = 0; j < kMaxDim; ++j) {
    if (j < trans_offset) {
      perm_compute[j] = j;
    } else {
      perm_compute[j] = perm_vec.at(j - trans_offset) + trans_offset;
    }
  }
  trans_output = trans_input.shuffle(perm_compute);
}

template <typename T_in, typename T_out>
uint32_t TraceV2CpuKernel::TraceV2Compute(CpuKernelContext &ctx) {
  T_in *input_addr = reinterpret_cast<T_in *>(ctx.Input(kIndexIn)->GetData());
  int64_t *offset_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexOffset)->GetData());
  int64_t offset = *offset_ptr;
  int64_t *axis1_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexAxis1)->GetData());
  int64_t axis1 = *axis1_ptr;
  int64_t *axis2_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexAxis2)->GetData());
  int64_t axis2 = *axis2_ptr;
  T_out *output_addr = reinterpret_cast<T_out *>(ctx.Output(kIndexOut)->GetData());

  std::vector<int64_t> shape_in = ctx.Input(kIndexIn)->GetTensorShape()->GetDimSizes();
  int64_t in_rank = static_cast<int64_t>(shape_in.size());
  if (in_rank < 2) {
    CUST_KERNEL_LOG_ERROR(ctx, "For [%s], dim of matrix a must greater or equal to 2, but got a at[%lld]-dimensional.",
                          kTraceV2, in_rank);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (axis1 < -in_rank || axis1 >= in_rank) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "For [%s], the value of input 'axis1' must in [-[%lld], [%lld]), but got 'axis1':[%lld].",
                          kTraceV2, in_rank, in_rank, axis1);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (axis2 < -in_rank || axis2 >= in_rank) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "For [%s], the value of input 'axis2' must in [-[%lld], [%lld]), but got 'axis2':[%lld].",
                          kTraceV2, in_rank, in_rank, axis2);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  axis1 = axis1 < 0 ? axis1 + in_rank : axis1;
  axis2 = axis2 < 0 ? axis2 + in_rank : axis2;
  if (axis1 == axis2) {
    CUST_KERNEL_LOG_ERROR(
      ctx, "For [%s], the value of 'axis1' and 'axis2' must be different, but got 'axis1':[%lld] and 'axis2'[%lld]",
      kTraceV2, axis1, axis2);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t in_size = std::accumulate(shape_in.begin(), shape_in.end(), 1, std::multiplies<int64_t>());
  int64_t mat_size = shape_in[axis1] * shape_in[axis2];
  int64_t mat_row_size = shape_in[axis2];
  int64_t mat_col_size = shape_in[axis1];
  int64_t batch_size;
  if (in_size == 0) {
    batch_size = 1;
  } else {
    batch_size = in_size / mat_size;
  }

  for (int64_t i = 0; i < batch_size; i++) {
    output_addr[i] = static_cast<T_out>(0);
  }

  T_in *trans_in_addr = static_cast<T_in *>(malloc(sizeof(T_in) * in_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, trans_in_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Tracev2] Malloc memory [trans_in_addr] failed!")
  transpose_step(input_addr, trans_in_addr, shape_in, axis1, axis2);
  T_out *cast_in = static_cast<T_out *>(malloc(sizeof(T_out) * in_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, cast_in, KERNEL_STATUS_PARAM_INVALID, "[Tracev2] Malloc memory [cast_in] failed!")
  for (int64_t i = 0; i < in_size; ++i) {
    Cast(trans_in_addr + i, cast_in + i);
  }
  for (int64_t i = 0; i < batch_size; i++) {
    T_out *cast_in_mat = cast_in + i * mat_size;
    output_addr[i] = static_cast<T_out>(0);
    int64_t row_idx;
    int64_t col_idx;
    if (offset > 0) {
      row_idx = 0;
      col_idx = offset;
    } else {
      col_idx = 0;
      row_idx = -offset;
    }
    while (row_idx < mat_col_size && col_idx < mat_row_size) {
      int64_t idx = row_idx * mat_row_size + col_idx;
      output_addr[i] += cast_in_mat[idx];
      row_idx++;
      col_idx++;
    }
  }
  free(cast_in);
  free(trans_in_addr);
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kTraceV2, TraceV2CpuKernel);
}  // namespace aicpu
