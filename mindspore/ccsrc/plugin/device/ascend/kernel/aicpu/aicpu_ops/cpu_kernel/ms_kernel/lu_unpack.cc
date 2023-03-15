/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "lu_unpack.h"
#include <string.h>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "cpu_context.h"
#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "cpu_tensor.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kOutputNum = 3;
const uint32_t kInputNum = 2;
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;
const uint32_t kSecondOutputIndex = 1;
const uint32_t kThirdOutputIndex = 2;
const int32_t kLuDataMinRank = 2;
const int32_t kLuPivotsMinRank = 2;
const int64_t kParallelBatchNum = 70;
const char *kLuUnpack = "LuUnpack";
}  // namespace
namespace aicpu {
template <typename T_data, typename T_pivots>
uint32_t LuUnpackCpuKernel::LuUnpack(CpuKernelContext &ctx, T_pivots *Lu_pivots_working_ptr, int64_t matrix_index,
                                     T_data *P_eye) {
  int32_t Lu_data_dims = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDims();
  int64_t Lu_data_dim1 = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSize(Lu_data_dims - 2);
  int64_t Lu_data_dim2 = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSize(Lu_data_dims - 1);
  int32_t Lu_pivots_dims = ctx.Input(kSecondInputIndex)->GetTensorShape()->GetDims();
  int64_t Lu_pivots_dim = ctx.Input(kSecondInputIndex)->GetTensorShape()->GetDimSize(Lu_pivots_dims - 1);
  int64_t matrix_width = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes()[Lu_data_dims - 2];
  int64_t matrix_height = ctx.Input(kFirstInputIndex)->GetTensorShape()->GetDimSizes()[Lu_data_dims - 1];
  int64_t pivots_stride = Lu_data_dim1 * Lu_data_dim1;
  int64_t L_stride = 0;
  int64_t U_stride = 0;
  if (Lu_data_dim1 > Lu_data_dim2) {
    L_stride = Lu_data_dim1 * Lu_data_dim2;
    U_stride = Lu_data_dim2 * Lu_data_dim2;
  } else {
    L_stride = Lu_data_dim1 * Lu_data_dim1;
    U_stride = Lu_data_dim1 * Lu_data_dim2;
  }
  int64_t matrix_size = matrix_width * matrix_height;
  using MatrixMap = Eigen::Map<Eigen::Matrix<T_data, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
  MatrixMap input(reinterpret_cast<T_data *>(ctx.Input(kFirstInputIndex)->GetData()) + matrix_index * matrix_size,
                  matrix_width, matrix_height);
  //  Triu
  if (matrix_width > matrix_height) {
    MatrixMap output2(reinterpret_cast<T_data *>(ctx.Output(kThirdOutputIndex)->GetData()) + matrix_index * U_stride,
                      matrix_height, matrix_height);
    T_data *MiddlePtr = new T_data[matrix_size];
    MatrixMap MiddleData(MiddlePtr, matrix_width, matrix_height);
    MiddleData = input.template triangularView<Eigen::Upper>();
    output2 = MiddleData.block(0, 0, matrix_height, matrix_height);
    delete[] MiddlePtr;
  } else {
    MatrixMap output2(reinterpret_cast<T_data *>(ctx.Output(kThirdOutputIndex)->GetData()) + matrix_index * U_stride,
                      matrix_width, matrix_height);
    output2 = input.template triangularView<Eigen::Upper>();
  }
  //  Tril
  if (matrix_height > matrix_width) {
    MatrixMap output1(reinterpret_cast<T_data *>(ctx.Output(kSecondOutputIndex)->GetData()) + matrix_index * L_stride,
                      matrix_width, matrix_width);
    T_data *MiddlePtr = new T_data[matrix_size];
    MatrixMap MiddleData(MiddlePtr, matrix_width, matrix_height);
    MiddleData = input.template triangularView<Eigen::UnitLower>();
    output1 = MiddleData.block(0, 0, matrix_width, matrix_width);
    delete[] MiddlePtr;
  } else {
    MatrixMap output1(reinterpret_cast<T_data *>(ctx.Output(kSecondOutputIndex)->GetData()) + matrix_index * L_stride,
                      matrix_width, matrix_height);
    output1 = input.template triangularView<Eigen::UnitLower>();
  }
  //  Swap
  std::vector<T_pivots> final_order;
  final_order.resize(Lu_data_dim1);
  for (int i = 0; i < Lu_data_dim1; i++) {
    final_order[i] = T_pivots(i);
  }
  for (T_pivots id = 0; id < Lu_pivots_dim; id++) {
    int64_t perm_id = 0;
    int64_t perm_pivots_id = 0;
    for (int64_t i = 0; i < Lu_data_dim1; i++) {
      if (id == final_order[i]) {
        perm_id = i;
      }
      if (!((*(Lu_pivots_working_ptr + id) <= Lu_data_dim1) && (*(Lu_pivots_working_ptr + id) >= 1))) {
        return KERNEL_STATUS_PARAM_INVALID;
      }
      if ((*(Lu_pivots_working_ptr + id) - 1) == final_order[i]) {
        perm_pivots_id = i;
      }
    }
    std::swap(final_order[perm_id], final_order[perm_pivots_id]);
  }
  //  Index_select
  auto output_y0 = reinterpret_cast<T_data *>(ctx.Output(kFirstOutputIndex)->GetData());
  int64_t indices_num = final_order.size();
  int64_t inner_size = Lu_data_dim1;
  int64_t slice_size = inner_size * sizeof(T_data);
  for (int64_t j = 0; j < indices_num; ++j) {
    auto params_idx = final_order[j] * inner_size;
    auto out_idx = j * inner_size;
    memcpy(output_y0 + matrix_index * pivots_stride + out_idx, P_eye + params_idx, slice_size);
  }
  return KERNEL_STATUS_OK;
}

template <typename T_data, typename T_pivots>
uint32_t LuUnpackCpuKernel::LuUnpackCompute(CpuKernelContext &ctx) {
  Tensor *input0_tensor = ctx.Input(kFirstInputIndex);
  Tensor *input1_tensor = ctx.Input(kSecondInputIndex);
  auto input_0_Shape = input0_tensor->GetTensorShape();
  auto input_1_Shape = input1_tensor->GetTensorShape();
  int32_t Lu_data_dims = input_0_Shape->GetDims();
  int64_t Lu_data_dim1 = input_0_Shape->GetDimSize(Lu_data_dims - 2);
  int64_t Lu_data_dim2 = input_0_Shape->GetDimSize(Lu_data_dims - 1);
  int32_t Lu_pivots_dims = input_1_Shape->GetDims();
  int64_t Lu_pivots_dim = input_1_Shape->GetDimSize(Lu_pivots_dims - 1);
  auto input_dim_size = input_0_Shape->GetDimSizes();
  auto input_x1 = reinterpret_cast<T_pivots *>(input1_tensor->GetData());

  int32_t block_size = Lu_data_dim1 * Lu_data_dim1;
  T_data *P_eye = new T_data[block_size]{};
  T_data num = static_cast<T_data>(1);
  for (int32_t i = 0; i < Lu_data_dim1; i++) {
    *(P_eye + (Lu_data_dim1 + 1) * i) = num;
  }
  uint32_t check_status = 0;
  int64_t Lu_data_stride = Lu_data_dim1 * Lu_data_dim2;
  int64_t Lu_pivots_stride = Lu_pivots_dim;
  int64_t batch_num = ctx.Input(0)->NumElements() / Lu_data_stride;
  if (batch_num < kParallelBatchNum || Lu_data_dims == kLuDataMinRank) {
    for (int64_t matrix_index = 0; matrix_index < batch_num; matrix_index++) {
      T_pivots *Lu_pivots_working_ptr = input_x1 + matrix_index * Lu_pivots_stride;
      check_status = LuUnpack(ctx, Lu_pivots_working_ptr, matrix_index, P_eye);
      if (check_status == KERNEL_STATUS_PARAM_INVALID) {
        return check_status;
      }
    }
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > batch_num) {
      max_core_num = batch_num;
    }
    uint32_t parallel_status = 0;
    auto sharder = [&](int64_t start, int64_t end) {
      for (int64_t matrix_index = start; matrix_index < end; matrix_index++) {
        T_pivots *Lu_pivots_working_ptr = input_x1 + matrix_index * Lu_pivots_stride;
        if (LuUnpack(ctx, Lu_pivots_working_ptr, matrix_index, P_eye) == KERNEL_STATUS_OK) {
          parallel_status = KERNEL_STATUS_OK;
        } else {
          parallel_status = KERNEL_STATUS_PARAM_INVALID;
          break;
        }
      }
    };
    if (max_core_num == 0) {
      KERNEL_LOG_ERROR("max_core_num could not be 0.");
    }
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, batch_num, batch_num / max_core_num, sharder),
                        "LuUnpack Compute failed.");
    if (parallel_status != KERNEL_STATUS_OK) {
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  delete[] P_eye;
  return KERNEL_STATUS_OK;
}

void LuUnpackCpuKernel::SetMap() {
  calls_[DT_INT8][DT_INT8] = LuUnpackCompute<int8_t, int8_t>;
  calls_[DT_INT8][DT_UINT8] = LuUnpackCompute<int8_t, uint8_t>;
  calls_[DT_INT8][DT_INT16] = LuUnpackCompute<int8_t, int16_t>;
  calls_[DT_INT8][DT_INT32] = LuUnpackCompute<int8_t, int32_t>;
  calls_[DT_INT8][DT_INT64] = LuUnpackCompute<int8_t, int64_t>;

  calls_[DT_INT16][DT_INT8] = LuUnpackCompute<int16_t, int8_t>;
  calls_[DT_INT16][DT_INT16] = LuUnpackCompute<int16_t, int16_t>;
  calls_[DT_INT16][DT_INT32] = LuUnpackCompute<int16_t, int32_t>;
  calls_[DT_INT16][DT_INT64] = LuUnpackCompute<int16_t, int64_t>;
  calls_[DT_INT16][DT_UINT8] = LuUnpackCompute<int16_t, uint8_t>;

  calls_[DT_INT32][DT_INT8] = LuUnpackCompute<int32_t, int8_t>;
  calls_[DT_INT32][DT_INT16] = LuUnpackCompute<int32_t, int16_t>;
  calls_[DT_INT32][DT_INT32] = LuUnpackCompute<int32_t, int32_t>;
  calls_[DT_INT32][DT_INT64] = LuUnpackCompute<int32_t, int64_t>;
  calls_[DT_INT32][DT_UINT8] = LuUnpackCompute<int32_t, uint8_t>;

  calls_[DT_INT64][DT_INT8] = LuUnpackCompute<int64_t, int8_t>;
  calls_[DT_INT64][DT_INT16] = LuUnpackCompute<int64_t, int16_t>;
  calls_[DT_INT64][DT_INT32] = LuUnpackCompute<int64_t, int32_t>;
  calls_[DT_INT64][DT_INT64] = LuUnpackCompute<int64_t, int64_t>;
  calls_[DT_INT64][DT_UINT8] = LuUnpackCompute<int64_t, uint8_t>;

  calls_[DT_FLOAT16][DT_INT8] = LuUnpackCompute<Eigen::half, int8_t>;
  calls_[DT_FLOAT16][DT_INT16] = LuUnpackCompute<Eigen::half, int16_t>;
  calls_[DT_FLOAT16][DT_INT32] = LuUnpackCompute<Eigen::half, int32_t>;
  calls_[DT_FLOAT16][DT_INT64] = LuUnpackCompute<Eigen::half, int64_t>;
  calls_[DT_FLOAT16][DT_UINT8] = LuUnpackCompute<Eigen::half, uint8_t>;

  calls_[DT_FLOAT][DT_INT8] = LuUnpackCompute<float, int8_t>;
  calls_[DT_FLOAT][DT_INT16] = LuUnpackCompute<float, int16_t>;
  calls_[DT_FLOAT][DT_INT32] = LuUnpackCompute<float, int32_t>;
  calls_[DT_FLOAT][DT_INT64] = LuUnpackCompute<float, int64_t>;
  calls_[DT_FLOAT][DT_UINT8] = LuUnpackCompute<float, uint8_t>;

  calls_[DT_DOUBLE][DT_INT8] = LuUnpackCompute<double, int8_t>;
  calls_[DT_DOUBLE][DT_INT16] = LuUnpackCompute<double, int16_t>;
  calls_[DT_DOUBLE][DT_INT32] = LuUnpackCompute<double, int32_t>;
  calls_[DT_DOUBLE][DT_INT64] = LuUnpackCompute<double, int64_t>;
  calls_[DT_DOUBLE][DT_UINT8] = LuUnpackCompute<double, uint8_t>;

  calls_[DT_UINT8][DT_INT8] = LuUnpackCompute<uint8_t, int8_t>;
  calls_[DT_UINT8][DT_INT16] = LuUnpackCompute<uint8_t, int16_t>;
  calls_[DT_UINT8][DT_INT32] = LuUnpackCompute<uint8_t, int32_t>;
  calls_[DT_UINT8][DT_INT64] = LuUnpackCompute<uint8_t, int64_t>;
  calls_[DT_UINT8][DT_UINT8] = LuUnpackCompute<uint8_t, uint8_t>;
}

void LuUnpackCpuKernel::SetOutputShape(CpuKernelContext &ctx) {
  Tensor *LU_data_ = ctx.Input(0);
  Tensor *output0 = ctx.Output(0);
  Tensor *output1 = ctx.Output(1);
  Tensor *output2 = ctx.Output(2);
  std::vector<int64_t> LU_data_shape = LU_data_->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> pivots_shape = LU_data_shape;
  std::vector<int64_t> L_shape = LU_data_shape;
  std::vector<int64_t> U_shape = LU_data_shape;
  int64_t Lu_data_dim1 = LU_data_shape.size() - 2;
  int64_t Lu_data_dim2 = LU_data_shape.size() - 1;
  if (LU_data_shape[Lu_data_dim1] != LU_data_shape[Lu_data_dim2]) {
    pivots_shape[Lu_data_dim2] = LU_data_shape[Lu_data_dim1];
  }
  if (LU_data_shape[Lu_data_dim1] < LU_data_shape[Lu_data_dim2]) {
    L_shape[Lu_data_dim2] = LU_data_shape[Lu_data_dim1];
  }
  if (LU_data_shape[Lu_data_dim1] > LU_data_shape[Lu_data_dim2]) {
    U_shape[Lu_data_dim1] = LU_data_shape[Lu_data_dim2];
  }

  output0->GetTensorShape()->SetDimSizes(pivots_shape);
  output1->GetTensorShape()->SetDimSizes(L_shape);
  output2->GetTensorShape()->SetDimSizes(U_shape);
}

uint32_t LuUnpackCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "LuUnpack check input and output number failed.");
  Tensor *LU_data_ = ctx.Input(0);
  Tensor *LU_pivots_ = ctx.Input(1);
  std::shared_ptr<TensorShape> LU_data_shape = LU_data_->GetTensorShape();
  std::shared_ptr<TensorShape> LU_pivots_shape = LU_pivots_->GetTensorShape();
  int32_t LU_data_rank = LU_data_shape->GetDims();
  if (LU_data_rank < kLuDataMinRank) {
    KERNEL_LOG_ERROR(
      "The input dim size of LU_data must be at least 2-D, "
      "while %d",
      LU_data_rank);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int32_t Lu_data_dims = LU_data_shape->GetDims();
  int64_t Lu_data_dim1 = LU_data_shape->GetDimSize(Lu_data_dims - 2);
  int64_t Lu_data_dim2 = LU_data_shape->GetDimSize(Lu_data_dims - 1);
  int32_t Lu_pivots_dims = LU_pivots_shape->GetDims();
  int64_t Lu_pivots_dim = LU_pivots_shape->GetDimSize(Lu_pivots_dims - 1);
  if (Lu_pivots_dim != std::min(Lu_data_dim1, Lu_data_dim2)) {
    KERNEL_LOG_ERROR(
      "The last dimension of LU_pivots must be the same as the minimum value "
      "of the last two dimensions of LU_data, "
      "but got The last dimension of LU_pivots [%d], the minimum value of "
      "the last two dimensions of LU_data: [%d]",
      Lu_pivots_dim, std::min(Lu_data_dim1, Lu_data_dim2));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (int32_t i = 0; i < Lu_pivots_dims - 1; i++) {
    if (LU_data_shape->GetDimSize(i) != LU_pivots_shape->GetDimSize(i)) {
      KERNEL_LOG_ERROR(
        " LU_data's batch dimensions does not match LU_pivots's batch "
        "dimensions.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  SetOutputShape(ctx);

  DataType LU_data_dtype = static_cast<DataType>(LU_data_->GetDataType());
  bool LU_data_dtype_flag = LU_data_dtype != DT_FLOAT16 && LU_data_dtype != DT_FLOAT && LU_data_dtype != DT_DOUBLE &&
                            LU_data_dtype != DT_INT8 && LU_data_dtype != DT_UINT8 && LU_data_dtype != DT_INT16 &&
                            LU_data_dtype != DT_INT32 && LU_data_dtype != DT_INT64;
  if (LU_data_dtype_flag) {
    KERNEL_LOG_ERROR(
      "Op LuUnpack first input LU_data_type's data type should be of the "
      "follows: "
      "DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, "
      "DT_FLOAT, DT_DOUBLE, "
      "but this type is [%s].",
      DTypeStr(LU_data_dtype).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  DataType LU_pivots_dtype = static_cast<DataType>(LU_pivots_->GetDataType());
  bool LU_pivots_dtype_flag = LU_pivots_dtype != DT_INT8 && LU_pivots_dtype != DT_UINT8 &&
                              LU_pivots_dtype != DT_INT16 && LU_pivots_dtype != DT_INT32 && LU_pivots_dtype != DT_INT64;
  if (LU_pivots_dtype_flag) {
    KERNEL_LOG_ERROR(
      "Op LuUnpack second input LU_pivots_type's data type should be of the "
      "follows: "
      "DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64, "
      "but this type is [%s].",
      DTypeStr(LU_pivots_dtype).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  SetMap();
  std::vector<DataType> LU_data_type_vec = {DT_INT8,  DT_UINT8,   DT_INT16, DT_INT32,
                                            DT_INT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE};
  std::vector<DataType> LU_pivots_type_vec = {DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64};
  for (uint64_t i = 0; i < LU_data_type_vec.size(); i++) {
    for (uint64_t j = 0; j < LU_pivots_type_vec.size(); j++) {
      if (LU_data_dtype == LU_data_type_vec[i] && LU_pivots_dtype == LU_pivots_type_vec[j]) {
        KERNEL_HANDLE_ERROR(calls_[LU_data_type_vec[i]][LU_pivots_type_vec[j]](ctx),
                            "The elements of LU_pivots must be greater than 1 "
                            "and be less than the size of LU_pivots's last dimension.");
      }
    }
  }
  calls_.clear();
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kLuUnpack, LuUnpackCpuKernel);
}  // namespace aicpu
