/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "bias_add_grad.h"

#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kBiasAddGrad = "BiasAddGrad";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 1;
}  // namespace
namespace aicpu {
uint32_t BiasAddGradCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "BiasAddGrad check input and output number failed.");
  Tensor *input = ctx.Input(kFirstInputIndex);
  if (input->GetDataSize() == 0) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_OK;
  }
  // choose compute function depend on dataType
  auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16:
      return BiasAddGradCompute<Eigen::half>(ctx);
    case DT_FLOAT:
      return BiasAddGradCompute<float>(ctx);
    case DT_DOUBLE:
      return BiasAddGradCompute<double>(ctx);
    case DT_INT8:
      return BiasAddGradCompute<int8_t>(ctx);
    case DT_INT16:
      return BiasAddGradCompute<int16_t>(ctx);
    case DT_INT32:
      return BiasAddGradCompute<int32_t>(ctx);
    case DT_INT64:
      return BiasAddGradCompute<int64_t>(ctx);
    case DT_UINT8:
      return BiasAddGradCompute<uint8_t>(ctx);
    case DT_UINT16:
      return BiasAddGradCompute<uint16_t>(ctx);
    case DT_UINT32:
      return BiasAddGradCompute<uint32_t>(ctx);
    case DT_UINT64:
      return BiasAddGradCompute<uint64_t>(ctx);
    case DT_COMPLEX64:
      return BiasAddGradCompute<std::complex<std::float_t>>(ctx);
    case DT_COMPLEX128:
      return BiasAddGradCompute<std::complex<std::double_t>>(ctx);
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T>
uint32_t BiasAddGradCpuKernel::BiasAddGradCompute(CpuKernelContext &ctx) {
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);

  KERNEL_LOG_DEBUG(
    "[DEBUG] [%s] Input data NumElements is [%llu]"
    "output data NumElements is [%llu].",
    ctx.GetOpType().c_str(), input->NumElements(), output->NumElements());
  KERNEL_LOG_DEBUG(
    "[DEBUG] [%s] Input data size is [%llu]"
    "output data size is [%llu].",
    ctx.GetOpType().c_str(), input->GetDataSize(), output->GetDataSize());

  auto input_shape = input->GetTensorShape();
  auto dims = input_shape->GetDimSizes();  // size: 4(NHWC/NCHW)
  auto attr_data_format = ctx.GetAttr("data_format");
  std::string str_format = (attr_data_format == nullptr) ? "NHWC" : (attr_data_format->GetString());

  // check input's dim
  if (input_shape->GetDims() != 4) {
    KERNEL_LOG_ERROR("Input's dim should be 4, but got [%d]", input_shape->GetDims());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // check input's format
  if (str_format != "NHWC" && str_format != "NCHW") {
    KERNEL_LOG_ERROR("Input's data format is invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto input_x = reinterpret_cast<T *>(input->GetData());
  auto output_y = reinterpret_cast<T *>(output->GetData());

  try {
    // init
    for (size_t i = 0; i < (size_t)output->NumElements(); i++) output_y[i] = (T).0f;

    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    KERNEL_LOG_DEBUG("[DEBUG] Cores Num: [%lld].", max_core_num);
    const size_t data_num = (size_t)input->NumElements();

    size_t step = 0;
    size_t length = 1;
    if (str_format == "NHWC") {
      step = dims[3];
      KERNEL_LOG_DEBUG("[DEBUG] data's format: NHWC");
    } else if (str_format == "NCHW") {
      step = dims[1];
      length = dims[2] * dims[3];
      KERNEL_LOG_DEBUG("[DEBUG] data's format: NCHW");
    }

    // run
    if (data_num >= 4 * 1024) {
      std::unordered_map<std::thread::id, T *> task_unit_res;
      std::mutex mtx;

      auto compute_task = [&](size_t start, size_t end) {
        std::thread::id tid = std::this_thread::get_id();

        mtx.lock();
        T *res;
        if (task_unit_res.find(tid) == task_unit_res.end()) {
          res = new T[step];
          for (size_t i = 0; i < step; i++) res[i] = (T).0f;
          task_unit_res[tid] = res;
        } else {
          res = task_unit_res[tid];
        }
        mtx.unlock();

        // compute unit
        for (size_t i = start; i < end; i++) {
          res[(i / length) % step] += input_x[i];
        }
      };
      CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, compute_task);
      for (auto itr = task_unit_res.begin(); itr != task_unit_res.end(); itr++) {
        T *unit_res = itr->second;
        for (size_t i = 0; i < step; i++) {
          output_y[i] += unit_res[i];
        }
        // free memory
        delete[] unit_res;
      }
    } else {
      for (size_t i = 0; i < data_num; i++) {
        output_y[(i / length) % step] += input_x[i];
      }
    }
  } catch (...) {
    KERNEL_LOG_ERROR("Compute Failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kBiasAddGrad, BiasAddGradCpuKernel);
}  // namespace aicpu
