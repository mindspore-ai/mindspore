/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "batchmatmul.h"

#include <complex>
#include "unsupported/Eigen/CXX11/Tensor"

#include "cpu_kernel_utils.h"
#include "utils/kernel_util.h"
#include "kernel_log.h"
#include "status.h"
#include <iostream>

using namespace std;

namespace {
const char *kBatchMatmul = "BatchMatMul";
const uint32_t kInputNum = 2;
const uint32_t kOutputNum = 1;
const int64_t kParallelDataNum = 1024;
}  // namespace

namespace aicpu {
template <typename T>
uint32_t BatchMatMulCpuKernel::DoCompute(CpuKernelContext &ctx) {
  auto input0_tensor = ctx.Input(0);
  auto input0_tensor_shape = input0_tensor->GetTensorShape();
  int32_t input0_tensor_dims = input0_tensor_shape->GetDims();
  KERNEL_CHECK_FALSE((input0_tensor_dims > 1), KERNEL_STATUS_PARAM_INVALID, "Input[x1] must be a matrix or higher.")

  auto input1_tensor = ctx.Input(1);
  auto input1_tensor_shape = input1_tensor->GetTensorShape();
  int32_t input1_tensor_dims = input1_tensor_shape->GetDims();
  KERNEL_CHECK_FALSE((input1_tensor_dims > 1), KERNEL_STATUS_PARAM_INVALID, "Input[x2] must be a matrix or higher.")

  auto output_tensor = ctx.Output(0);
  DataType input0_data_type = input0_tensor->GetDataType();
  DataType input1_data_type = input1_tensor->GetDataType();
  DataType output_data_type = output_tensor->GetDataType();
  KERNEL_CHECK_FALSE((input0_data_type == input1_data_type), KERNEL_STATUS_PARAM_INVALID,
                     "Input[x1] data type[%s] and input[x2] data type[%s] must be same.",
                     DTypeStr(input0_data_type).c_str(), DTypeStr(input1_data_type).c_str())
  KERNEL_CHECK_FALSE((input0_data_type == output_data_type), KERNEL_STATUS_PARAM_INVALID,
                     "Input data type[%s] and output data type[%s] must be same.", DTypeStr(input0_data_type).c_str(),
                     DTypeStr(output_data_type).c_str())
  bool adj_x = false;
  bool adj_y = false;
  auto adj_x1 = ctx.GetAttr("adj_x1");
  auto adj_x2 = ctx.GetAttr("adj_x2");
  if (adj_x1 != nullptr) {
    adj_x = adj_x1->GetBool();
  }
  if (adj_x2 != nullptr) {
    adj_y = adj_x2->GetBool();
  }
  KERNEL_LOG_DEBUG(
    "%s Attr[adj_x1] value[%d], "
    "Attr[adj_x2] value[%d].",
    kBatchMatmul, adj_x, adj_y);

  int32_t x1_dim = adj_x ? input0_tensor_dims - 2 : input0_tensor_dims - 1;
  int32_t x2_dim = adj_y ? input1_tensor_dims - 1 : input1_tensor_dims - 2;
  KERNEL_CHECK_FALSE((input0_tensor_shape->GetDimSize(x1_dim) == input1_tensor_shape->GetDimSize(x2_dim)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "Matrix size incompatible, input[x1] dim[%d] value[%lld], "
                     "input[x2] dim[%d] value[%lld]",
                     x1_dim, input0_tensor_shape->GetDimSize(x1_dim), x2_dim, input1_tensor_shape->GetDimSize(x2_dim))
  KERNEL_CHECK_FALSE((input0_tensor_dims == input1_tensor_dims), KERNEL_STATUS_PARAM_INVALID,
                     "input0_tensor_dims value[%d] is not equal to "
                     "input1_tensor_dims value[%d]",
                     input0_tensor_dims, input1_tensor_dims)

  auto input0_shape = input0_tensor_shape->GetDimSizes();
  auto input1_shape = input1_tensor_shape->GetDimSizes();
  auto output_shape = output_tensor->GetTensorShape()->GetDimSizes();

  for (int32_t i = 0; i < input0_tensor_dims - 2; i++) {
    KERNEL_CHECK_FALSE((input0_shape[i] == input1_shape[i]), KERNEL_STATUS_PARAM_INVALID,
                       "input0_shape dim[%d] value[%lld] is not equal to "
                       "input1_shape dim[%d] value[%lld]",
                       i, input0_shape[i], i, input1_shape[i])

    KERNEL_CHECK_FALSE((input0_shape[i] == output_shape[i]), KERNEL_STATUS_PARAM_INVALID,
                       "input0_shape dim[%d] value[%lld] is not equal to "
                       "output_shape dim[%d] value[%lld]",
                       i, input0_shape[i], i, output_shape[i])
  }

  int32_t map1_l = input0_shape[input0_tensor_dims - 2];
  int32_t map1_r = input0_shape[input0_tensor_dims - 1];

  int32_t map2_l = input1_shape[input1_tensor_dims - 2];
  int32_t map2_r = input1_shape[input1_tensor_dims - 1];

  int32_t rows_dim = adj_x ? input0_tensor_dims - 1 : input0_tensor_dims - 2;
  int32_t cols_dim = adj_y ? input1_tensor_dims - 2 : input1_tensor_dims - 1;

  int32_t num_rows = input0_tensor_shape->GetDimSize(rows_dim);
  int32_t num_cols = input1_tensor_shape->GetDimSize(cols_dim);

  uint64_t num_element = output_tensor->NumElements();
  uint64_t num_batches = num_element / (num_rows * num_cols);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> map1(map1_l, map1_r);
  auto *input0_data = reinterpret_cast<T *>(input0_tensor->GetData());
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> map2(map2_l, map2_r);
  auto *input1_data = reinterpret_cast<T *>(input1_tensor->GetData());
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> map_output(num_rows, num_cols);
  auto *output_data = reinterpret_cast<T *>(output_tensor->GetData());
  uint64_t input0_size = input0_tensor->GetDataSize();
  if (input0_size >= kParallelDataNum) {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > num_batches) {
      max_core_num = num_batches;
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> map1[num_batches];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> map2[num_batches];
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> map_output[num_batches];
    auto shared_batchmatmul = [&](int64_t start, int64_t end) {
      for (int64_t batch = start; batch < end; ++batch) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix1(map1_l, map1_r);
        map1[batch].resize(map1_l, map1_r);
        for (int64_t i = 0; i < map1_l; i++) {
          for (int64_t j = 0; j < map1_r; j++) {
            map1[batch](i, j) = input0_data[batch * map1_l * map1_r + i * map1_r + j];
          }
        }
        map2[batch].resize(map2_l, map2_r);
        for (int64_t i = 0; i < map2_l; i++) {
          for (int64_t j = 0; j < map2_r; j++) {
            map2[batch](i, j) = input1_data[batch * map2_l * map2_r + i * map2_r + j];
          }
        }
        if (adj_x) {
          if (adj_y) {
            map_output[batch] = map1[batch].adjoint() * map2[batch].adjoint();
          } else {
            map_output[batch] = map1[batch].adjoint() * map2[batch];
          }
        } else {
          if (adj_y) {
            map_output[batch] = map1[batch] * map2[batch].adjoint();
          } else {
            map_output[batch] = map1[batch] * map2[batch];
          }
        }
        map_output[batch].resize(num_rows, num_cols);
        for (int64_t i = 0; i < num_rows; ++i) {
          for (int64_t j = 0; j < num_cols; ++j) {
            output_data[batch * num_rows * num_cols + i * num_cols + j] = map_output[batch](i, j);
          }
        }
      }
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, num_batches, num_batches / max_core_num, shared_batchmatmul),
                        "BatchMatMul Compute failed.");
  } else {
    for (uint64_t batch = 0; batch < num_batches; ++batch) {
      for (int64_t i = 0; i < map1_l; i++) {
        for (int64_t j = 0; j < map1_r; j++) {
          map1(i, j) = input0_data[batch * map1_l * map1_r + i * map1_r + j];
        }
      }
      for (int64_t i = 0; i < map2_l; i++) {
        for (int64_t j = 0; j < map2_r; j++) {
          map2(i, j) = input1_data[batch * map2_l * map2_r + i * map2_r + j];
        }
      }
      if (adj_x) {
        if (adj_y) {
          map_output = map1.adjoint() * map2.adjoint();
        } else {
          map_output = map1.adjoint() * map2;
        }
      } else {
        if (adj_y) {
          map_output = map1 * map2.adjoint();
        } else {
          map_output = map1 * map2;
        }
      }
      for (int64_t i = 0; i < num_rows; ++i) {
        for (int64_t j = 0; j < num_cols; ++j) {
          output_data[batch * num_rows * num_cols + i * num_cols + j] = map_output(i, j);
        }
      }
    }
  }
  return KERNEL_STATUS_OK;
}

uint32_t BatchMatMulCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check BatchMatMul params failed.");
  DataType input0_data_type = ctx.Input(0)->GetDataType();
  KERNEL_LOG_DEBUG("%s op input[x1] data type is [%s].", kBatchMatmul, DTypeStr(input0_data_type).c_str());
  uint32_t ret = KERNEL_STATUS_OK;
  switch (input0_data_type) {
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    case DT_FLOAT16:
      ret = DoCompute<Eigen::half>(ctx);
      break;
    case DT_INT16:
      ret = DoCompute<int16_t>(ctx);
      break;
    case DT_INT32:
      ret = DoCompute<int32_t>(ctx);
      break;
    case DT_INT64:
      ret = DoCompute<int64_t>(ctx);
      break;
    case DT_UINT16:
      ret = DoCompute<uint16_t>(ctx);
      break;
    case DT_UINT32:
      ret = DoCompute<uint32_t>(ctx);
      break;
    case DT_UINT64:
      ret = DoCompute<uint64_t>(ctx);
      break;
    case DT_COMPLEX64:
      ret = DoCompute<std::complex<std::float_t>>(ctx);
      break;
    case DT_COMPLEX128:
      ret = DoCompute<std::complex<std::double_t>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input[x1] data type[%s]", DTypeStr(input0_data_type).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
  }
  return ret;
}
REGISTER_CPU_KERNEL(kBatchMatmul, BatchMatMulCpuKernel);
}  // namespace aicpu
