/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/ms_kernel/matrix_set_diag_v3.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "common/kernel_log.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char *MATRIX_SET_DIAG_V3 = "MatrixSetDiagV3";
const int64_t kParallelDataNum = 64 * 1024;
}  // namespace

namespace aicpu {
uint32_t MatrixSetDiagV3CpuKernel::CheckParam(const CpuKernelContext &ctx) {
  // check params
  Tensor *input_tensor = ctx.Input(0);
  Tensor *diagonal_tensor = ctx.Input(1);
  Tensor *output_tensor = ctx.Output(0);
  auto input_dtype = input_tensor->GetDataType();
  auto diagonal_dtype = diagonal_tensor->GetDataType();
  auto output_dtype = output_tensor->GetDataType();
  std::string align = "RIGHT_LEFT";
  AttrValue *attr_align = ctx.GetAttr("align");
  if (attr_align != NULL) {
    align = attr_align->GetString();
  }
  // check tensor_type
  KERNEL_CHECK_FALSE((input_dtype == diagonal_dtype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input_dtype [%d] need be same with "
                     "diagonal_type [%d].",
                     input_dtype, diagonal_dtype);
  KERNEL_CHECK_FALSE((input_dtype == output_dtype), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input_dtype [%d] need be same with "
                     "output_dtype [%d].",
                     input_dtype, output_dtype);
  // check align
  KERNEL_CHECK_FALSE(
    (align == "" || align == "RIGHT_LEFT" || align == "RIGHT_RIGHT" || align == "LEFT_LEFT" || align == "LEFT_RIGHT"),
    KERNEL_STATUS_PARAM_INVALID,
    "Attr 'align' of 'MatrixSetDiagV3' is not in: 'LEFT_RIGHT', "
    "'RIGHT_LEFT', 'LEFT_LEFT', 'RIGHT_RIGHT'.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MatrixSetDiagV3CpuKernel::DoCompute(const CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  auto input_shape = input_tensor->GetTensorShape();
  int64_t input_dims = input_shape->GetDims();
  int64_t input_cloumns = input_shape->GetDimSize(input_dims - 1);
  int64_t input_rows = input_shape->GetDimSize(input_dims - 2);
  auto input_data = reinterpret_cast<T *>(input_tensor->GetData());
  int64_t input_numelements = input_tensor->NumElements();

  Tensor *diagonal_tensor = ctx.Input(1);
  auto diagonal_shape = diagonal_tensor->GetTensorShape();
  int64_t diagonal_dims = diagonal_shape->GetDims();
  int64_t diagonal_cloumns = diagonal_shape->GetDimSize(diagonal_dims - 1);
  int64_t diagonal_rows = 1;
  if (diagonal_dims > 1) diagonal_rows = diagonal_shape->GetDimSize(diagonal_dims - 2);
  auto diagonal_data = reinterpret_cast<T *>(diagonal_tensor->GetData());

  Tensor *k_tensor = ctx.Input(2);
  int64_t k_len = k_tensor->NumElements();
  int64_t k_lower = 0;
  int64_t k_upper = 0;
  auto k_Data = reinterpret_cast<int32_t *>(k_tensor->GetData());
  KERNEL_CHECK_FALSE((k_len == 1 || k_len == 2), KERNEL_STATUS_PARAM_INVALID,
                     "tensor_k dims size  must <= 2,got size [%lld].", k_len);
  k_lower = k_Data[0];
  k_upper = k_Data[0];
  if (k_len == 2) k_upper = k_Data[1];
  KERNEL_CHECK_FALSE((k_lower <= k_upper), KERNEL_STATUS_PARAM_INVALID, " k[0] must not be larger than k[1] .");

  std::string align = "RIGHT_LEFT";
  AttrValue *attr_align = ctx.GetAttr("align");
  if (attr_align != NULL) {
    align = attr_align->GetString();
  }

  Tensor *output_tensor = ctx.Output(0);
  auto output_data = reinterpret_cast<T *>(output_tensor->GetData());

  const int64_t zero = 0;
  int64_t max_diag_len = std::min(input_rows + std::min(k_upper, zero), input_cloumns + std::min(-k_lower, zero));

  for (int64_t i = 0; i < input_dims - 2; ++i) {
    KERNEL_CHECK_FALSE((input_shape->GetDimSize(i) == diagonal_shape->GetDimSize(i)), KERNEL_STATUS_PARAM_INVALID,
                       "diagonal_shape has incorrect value of elements[%lld] got %lld, should "
                       "be %lld",
                       i, diagonal_shape->GetDimSize(i), input_shape->GetDimSize(i));
  }

  if (k_upper != k_lower) {
    KERNEL_CHECK_FALSE((diagonal_rows == (k_upper - k_lower + 1)), KERNEL_STATUS_PARAM_INVALID,
                       "diagonal_shape has incorrect value of elements[%lld] got %lld, should "
                       "be %lld",
                       diagonal_dims - 2, diagonal_rows, (k_upper - k_lower + 1));
  }
  KERNEL_CHECK_FALSE((max_diag_len == diagonal_cloumns), KERNEL_STATUS_PARAM_INVALID,
                     "diagonal_shape has incorrect value of elements[%lld] got "
                     "%lld, should be %lld",
                     diagonal_dims - 1, diagonal_cloumns, max_diag_len);

  // Fill the output diagonal.64K前单核计算，64K后所有核计算
  uint64_t input_size = input_tensor->GetDataSize();
  if (input_size < kParallelDataNum) {
    if (k_len == 1 || (k_len == 2 && k_lower == k_upper)) {
      for (int64_t elem = 0; elem < input_numelements; ++elem) {
        int64_t t = elem % (input_rows * input_cloumns);
        int64_t index = elem / (input_rows * input_cloumns);
        int64_t m = t / input_cloumns;
        int64_t n = t % input_cloumns;
        int64_t x = n - std::max(k_upper, zero);
        if (n - m == k_upper)
          output_data[elem] = diagonal_data[index * diagonal_cloumns + x];
        else
          output_data[elem] = input_data[elem];
      }
    } else {
      for (int64_t elem = 0; elem < input_numelements; ++elem) {
        int64_t t = elem % (input_rows * input_cloumns);
        int64_t index = elem / (input_rows * input_cloumns);
        int64_t m = t / input_cloumns;
        int64_t n = t % input_cloumns;
        int64_t d = n - m;
        if (d >= k_lower && d <= k_upper) {
          int64_t x = k_upper - d;
          int64_t offset = 0;
          if (((align == "RIGHT_LEFT" || align == "RIGHT_RIGHT") && d >= 0) ||
              ((align == "LEFT_RIGHT" || align == "RIGHT_RIGHT") && d <= 0)) {
            offset = max_diag_len - std::min(input_cloumns - std::max(d, zero), input_rows + std::min(d, zero));
          }
          int64_t y = n - std::max(d, zero) + offset;
          output_data[elem] = diagonal_data[index * diagonal_rows * diagonal_cloumns + x * diagonal_cloumns + y];
        } else {
          output_data[elem] = input_data[elem];
        }
      }
    }
  } else {
    uint32_t min_core_num = 1;
    // 使用CpuKernelUtils::GetCPUNum接口获取AI CPU的核数
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > input_numelements) {
      max_core_num = input_numelements;
    }
    auto sharder_matrix_set_diag_v3 = [&](int64_t start, int64_t end) {
      if (k_len == 1 || (k_len == 2 && k_lower == k_upper)) {
        for (int64_t elem = start; elem < end; ++elem) {
          int64_t t = elem % (input_rows * input_cloumns);
          int64_t index = elem / (input_rows * input_cloumns);
          int64_t m = t / input_cloumns;
          int64_t n = t % input_cloumns;
          int64_t x = n - std::max(k_upper, zero);
          if (n - m == k_upper)
            output_data[elem] = diagonal_data[index * diagonal_cloumns + x];
          else
            output_data[elem] = input_data[elem];
        }
      } else {
        for (int64_t elem = start; elem < end; ++elem) {
          int64_t t = elem % (input_rows * input_cloumns);
          int64_t index = elem / (input_rows * input_cloumns);
          int64_t m = t / input_cloumns;
          int64_t n = t % input_cloumns;
          int64_t d = n - m;
          if (d >= k_lower && d <= k_upper) {
            int64_t x = k_upper - d;
            int64_t offset = 0;
            if (((align == "RIGHT_LEFT" || align == "RIGHT_RIGHT") && d >= 0) ||
                ((align == "LEFT_RIGHT" || align == "RIGHT_RIGHT") && d <= 0)) {
              offset = max_diag_len - std::min(input_cloumns - std::max(d, zero), input_rows + std::min(d, zero));
            }
            int64_t y = n - std::max(d, zero) + offset;
            output_data[elem] = diagonal_data[index * diagonal_rows * diagonal_cloumns + x * diagonal_cloumns + y];
          } else {
            output_data[elem] = input_data[elem];
          }
        }
      }
    };
    CpuKernelUtils::ParallelFor(ctx, input_numelements, input_numelements / max_core_num, sharder_matrix_set_diag_v3);
  }
  return KERNEL_STATUS_OK;
}

uint32_t MatrixSetDiagV3CpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "check input and output number failed.");
  KERNEL_CHECK_FALSE((CheckParam(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                     "The params in MatrixSetDiagV3 is error, CheckParam failed.");
  uint32_t ret = KERNEL_STATUS_OK;
  DataType dt = static_cast<DataType>(input_tensor->GetDataType());
  switch (dt) {
    case DT_INT8:
      ret = DoCompute<int8_t>(ctx);
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
    case DT_UINT8:
      ret = DoCompute<uint8_t>(ctx);
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
    case DT_DOUBLE:
      ret = DoCompute<double>(ctx);
      break;
    case DT_FLOAT:
      ret = DoCompute<float>(ctx);
      break;
    case DT_COMPLEX128:
      ret = DoCompute<std::complex<std::double_t>>(ctx);
      break;
    case DT_FLOAT16:
      ret = DoCompute<Eigen::half>(ctx);
      break;
    case DT_COMPLEX64:
      ret = DoCompute<std::complex<std::float_t>>(ctx);
      break;
    default:
      KERNEL_LOG_ERROR("Unsupported input data type[%s]", DTypeStr(dt).c_str());
      ret = KERNEL_STATUS_PARAM_INVALID;
      break;
  }
  return ret;
}

REGISTER_CPU_KERNEL(MATRIX_SET_DIAG_V3, MatrixSetDiagV3CpuKernel);
}  // namespace aicpu
