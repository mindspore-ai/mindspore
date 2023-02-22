/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
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
#include "bias_add.h"

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "iomanip"
#include "iostream"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kBiasAdd = "BiasAdd";
}
namespace aicpu {
uint32_t BiasAddCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalMathCheck(ctx) != KERNEL_STATUS_OK) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  Tensor *input_1 = ctx.Input(kSecondInputIndex);
  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  auto input_shape_X = input_0->GetTensorShape();
  auto input_shape_B = input_1->GetTensorShape();
  if (input_shape_B->GetDims() != 1) {
    KERNEL_LOG_ERROR("bias should be 1-D.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input_shape_X->GetDims() < 2) {
    KERNEL_LOG_ERROR("Input tensor must be at least 2D");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto attr_data_format = ctx.GetAttr("data_format");
  std::string str_format = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());
  int32_t shapesize = input_shape_X->GetDims();
  int32_t xsize = shapesize - 1;
  if (str_format == "NHWC") {
    if (input_shape_B->GetDimSize(0) != input_shape_X->GetDimSize(xsize)) {
      KERNEL_LOG_ERROR(
        "Must provide as many biases as the last dimension of the input "
        "tensor");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else if (str_format == "NCHW") {
    if (input_shape_B->GetDimSize(0) != input_shape_X->GetDimSize(1)) {
      KERNEL_LOG_ERROR(
        "Must provide as many biases as the last dimension of the input "
        "tensor");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  } else if (str_format == "NCDHW") {
    KERNEL_LOG_ERROR("For BiasAdd, 'data_format' should be `NHWC` or `NCHW`, but got [%s].", str_format.c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  switch (data_type) {
    case DT_FLOAT16:
      return BiasAddCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return BiasAddCompute<float>(ctx);
    case DT_DOUBLE:
      return BiasAddCompute<double>(ctx);
    case DT_INT8:
      return BiasAddCompute<int8_t>(ctx);
    case DT_INT16:
      return BiasAddCompute<int16_t>(ctx);
    case DT_INT32:
      return BiasAddCompute<int32_t>(ctx);
    case DT_INT64:
      return BiasAddCompute<int64_t>(ctx);
    case DT_UINT8:
      return BiasAddCompute<uint8_t>(ctx);
    case DT_UINT16:
      return BiasAddCompute<uint16_t>(ctx);
    case DT_UINT32:
      return BiasAddCompute<uint32_t>(ctx);
    case DT_UINT64:
      return BiasAddCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return BiasAddCompute<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return BiasAddCompute<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t BiasAddCpuKernel::BiasAddCompute(CpuKernelContext &ctx) {
  BCalcInfo calc_info;
  calc_info.input_0 = ctx.Input(kFirstInputIndex);
  calc_info.input_1 = ctx.Input(kSecondInputIndex);
  calc_info.output = ctx.Output(kFirstOutputIndex);
  auto attr_data_format = ctx.GetAttr("data_format");
  std::string str_format = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());
  auto dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  auto input_x = reinterpret_cast<T *>(calc_info.input_0->GetData());
  auto input_bias = reinterpret_cast<T *>(calc_info.input_1->GetData());
  auto output_y = reinterpret_cast<T *>(calc_info.output->GetData());
  int32_t shapesize = ctx.Input(0)->GetTensorShape()->GetDims();
  int64_t size = calc_info.input_0->NumElements();
  const int64_t kParallelDataNum = 32 * 1024;
  const int64_t kParallelDataNumMid = 256 * 1024;
  for (int64_t i = 0; i < size; i++) {
    output_y[i] = (T).0f;
  }
  if (str_format == "NCHW") {
    int32_t step = size / dims[0];
    int32_t area = step / dims[1];

    if (size >= kParallelDataNum) {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);

      if (size <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
      }

      if (max_core_num > size) {
        max_core_num = size;
      }

      auto sharder_biadadd = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
          output_y[i] = input_x[i] + input_bias[i % step / area];
        }
      };

      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, size, size / max_core_num, sharder_biadadd),
                          "Biasadd Compute failed.");
    } else {
      for (int64_t i = 0; i < size; i++) {
        output_y[i] = input_x[i] + input_bias[i % step / area];
      }
    }
    return KERNEL_STATUS_OK;
  } else if (str_format == "NHWC") {
    int64_t last = dims[shapesize - 1];
    if (size >= kParallelDataNum) {
      uint32_t min_core_num = 1;
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
      if (size <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, 4U);  // up to 4 cpu cores
      }
      if (max_core_num > size) {
        max_core_num = size;
      }
      auto sharder_biadadd = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; i++) {
          output_y[i] = input_x[i] + input_bias[i % last];
        }
      };
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, size, size / max_core_num, sharder_biadadd),
                          "Biasadd Compute failed.");
    } else {
      for (int64_t i = 0; i < size; i++) {
        output_y[i] = input_x[i] + input_bias[i % last];
      }
    }
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kBiasAdd, BiasAddCpuKernel);
}  // namespace aicpu