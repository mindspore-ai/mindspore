/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/expand_dims_kernels.h"
#include <memory.h>
#include <map>
#include <thread>
#include "ops/expand_dims.h"
#include "./kernel_log.h"
#include "./kernel_errcode.h"
#include "proto/node_def.pb.h"
#include "common/tensor.h"
#include "proto/attr.pb.h"
#include "aicpu_sharder/aicpu_sharder.h"
namespace aicpu {
namespace dataset {
uint32_t ExpandDimsKernel::DoCompute() {
  size_t type_size = GetDataTypeSize(matrix_info_.matrix_type);
  if (type_size < 1) {
    AICPU_LOGE("don't support input tensor types");
    return kAicpuKernelStateFailed;
  }
  int ret = memcpy_s(reinterpret_cast<void *>(io_addrs_[1]), input_size_ * type_size,
                     reinterpret_cast<void *>(io_addrs_[0]), input_size_ * type_size);
  if (ret < 0) {
    return kAicpuKernelStateFailed;
  }
  return kAicpuKernelStateSucess;
}

uint32_t ExpandDimsKernel::ParseKernelParam() {
  AICPU_LOGI("aicpu ExpandDimsKernel");

  aicpuops::Tensor input_tensor = node_def_.inputs(0);
  aicpuops::TensorShape input_shape = input_tensor.tensor_shape();
  input_size_ = 1;

  matrix_info_.matrix_type = static_cast<::aicpuops::DataType>(input_tensor.tensor_type());
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    matrix_info_.matrix_shape.push_back(input_shape.dim(i).size());
    input_size_ *= input_shape.dim(i).size();
  }

  return kAicpuKernelStateSucess;
}
}  // namespace dataset
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t ExpandDims(void *param) {
  aicpu::dataset::ExpandDimsKernel expandDimsKernel;
  return expandDimsKernel.Compute(param);
}
}
