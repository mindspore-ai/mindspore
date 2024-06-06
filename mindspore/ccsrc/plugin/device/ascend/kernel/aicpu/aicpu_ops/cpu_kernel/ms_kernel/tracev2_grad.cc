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

#include "cpu_kernel/ms_kernel/tracev2_grad.h"
#include "Eigen/Core"
#include "securec/include/securec.h"
#include "context/inc/cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "utils/eigen_tensor.h"

namespace {
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 5;
constexpr size_t kIndexDOut = 0;
constexpr size_t kIndexInShape = 1;
constexpr size_t kIndexOffset = 2;
constexpr size_t kIndexAxis1 = 3;
constexpr size_t kIndexAxis2 = 4;
constexpr size_t kIndexDin = 0;
constexpr size_t kMaxDim = 8;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
using float16 = Eigen::half;
using bfloat16 = Eigen::bfloat16;
const char *kTraceV2Grad = "TraceV2Grad";

}  // namespace

// 定义命名空间aicpu
namespace aicpu {
// 实现自定义算子类的Compute函数
uint32_t TraceV2GradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputNum, kOutputNum),
                           "Tracev2grad check input and output number failed.");
  DataType data_type = ctx.Input(kIndexDOut)->GetDataType();
  uint32_t ret = KERNEL_STATUS_OK;
  switch (data_type) {
    case DT_FLOAT:
      ret = TraceV2GradCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = TraceV2GradCompute<double>(ctx);
      break;
    case DT_FLOAT16:
      ret = TraceV2GradCompute<float16>(ctx);
      break;
    case DT_INT8:
      ret = TraceV2GradCompute<int8_t>(ctx);
      break;
    case DT_INT16:
      ret = TraceV2GradCompute<int16_t>(ctx);
      break;
    case DT_INT32:
      ret = TraceV2GradCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      ret = TraceV2GradCompute<int64_t>(ctx);
      break;
    case DT_UINT8:
      ret = TraceV2GradCompute<uint8_t>(ctx);
      break;
    case DT_UINT16:
      ret = TraceV2GradCompute<uint16_t>(ctx);
      break;
    case DT_UINT32:
      ret = TraceV2GradCompute<uint32_t>(ctx);
      break;
    case DT_UINT64:
      ret = TraceV2GradCompute<uint64_t>(ctx);
      break;
    case DT_COMPLEX64:
      ret = TraceV2GradCompute<complex64>(ctx);
      break;
    case DT_COMPLEX128:
      ret = TraceV2GradCompute<complex128>(ctx);
      break;
    case DT_BOOL:
      ret = TraceV2GradCompute<bool>(ctx);
      break;
    case DT_BFLOAT16:
      ret = TraceV2GradCompute<bfloat16>(ctx);
      break;
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "TraceV2Grad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}

template <typename T>
void transpose_step(T *in_addr, T *out_addr, int64_t *shape_out, int64_t in_rank, int64_t axis1, int64_t axis2) {
  size_t trans_offset = kMaxDim - in_rank;
  std::vector<int64_t> shape_in;
  std::vector<int64_t> perm_vec;
  std::vector<int64_t> trans_in_shape(kMaxDim, 1);
  std::vector<int64_t> trans_out_shape(kMaxDim, 1);
  int64_t idx = 0;
  for (int64_t i = 0; i < in_rank; i++) {
    if (i != axis1 && i != axis2) {
      shape_in.emplace_back(shape_out[i]);
      perm_vec.emplace_back(idx);
      idx += 1;
    } else if (i == axis1) {
      perm_vec.emplace_back(in_rank - 2);
    } else {
      perm_vec.emplace_back(in_rank - 1);
    }
  }
  shape_in.emplace_back(shape_out[axis1]);
  shape_in.emplace_back(shape_out[axis2]);

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

template <typename T>
uint32_t TraceV2GradCpuKernel::TraceV2GradCompute(CpuKernelContext &ctx) {
  T *dout_addr = reinterpret_cast<T *>(ctx.Input(kIndexDOut)->GetData());
  int64_t *input_shape = reinterpret_cast<int64_t *>(ctx.Input(kIndexInShape)->GetData());
  int64_t *offset_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexOffset)->GetData());
  int64_t offset = *offset_ptr;
  int64_t *axis1_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexAxis1)->GetData());
  int64_t axis1 = *axis1_ptr;
  int64_t *axis2_ptr = reinterpret_cast<int64_t *>(ctx.Input(kIndexAxis2)->GetData());
  int64_t axis2 = *axis2_ptr;
  T *din_addr = reinterpret_cast<T *>(ctx.Output(kIndexDin)->GetData());

  std::vector<int64_t> dout_shape = ctx.Input(kIndexDOut)->GetTensorShape()->GetDimSizes();
  int64_t dout_rank = static_cast<int64_t>(dout_shape.size());
  int64_t din_rank = dout_rank + 2;
  axis1 = axis1 < 0 ? axis1 + din_rank : axis1;
  axis2 = axis2 < 0 ? axis2 + din_rank : axis2;
  if (dout_rank == 1 && dout_shape[0] < 2) din_rank = 2;
  int64_t mat_size = input_shape[axis1] * input_shape[axis2];
  int64_t mat_row_size = input_shape[axis2];
  int64_t mat_col_size = input_shape[axis1];
  int64_t batch_size = 1;
  for (int64_t i = 0; i < din_rank; i++) {
    if (i != axis1 && i != axis2) {
      batch_size *= input_shape[i];
    }
  }
  int64_t din_size = batch_size * mat_size;

  T *trans_din_addr = static_cast<T *>(malloc(sizeof(T) * din_size));
  CUST_KERNEL_CHECK_NULLPTR(ctx, trans_din_addr, KERNEL_STATUS_PARAM_INVALID,
                            "[Tracev2Grad] Malloc memory [trans_din_addr] failed!")
  auto ret = memset_s(trans_din_addr, din_size * sizeof(T), 0x00, din_size * sizeof(T));
  CUST_KERNEL_CHECK_FALSE(ctx, (ret == EOK), KERNEL_STATUS_PARAM_INVALID, "TraceV2 memset_s failed.");

  for (int64_t i = 0; i < batch_size; i++) {
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
      int64_t idx = row_idx * mat_row_size + col_idx + i * mat_size;
      trans_din_addr[idx] = dout_addr[i];
      row_idx++;
      col_idx++;
    }
  }
  transpose_step(trans_din_addr, din_addr, input_shape, din_rank, axis1, axis2);
  free(trans_din_addr);
  return KERNEL_STATUS_OK;
}
REGISTER_MS_CPU_KERNEL(kTraceV2Grad, TraceV2GradCpuKernel);
}  // namespace aicpu
