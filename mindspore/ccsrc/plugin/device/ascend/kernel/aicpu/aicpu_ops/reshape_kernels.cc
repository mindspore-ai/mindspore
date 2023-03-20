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

#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/reshape_kernels.h"
#include <memory.h>
#include <map>
#include <thread>
#include <atomic>
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "ops/reshape.h"
#include "./kernel_log.h"
#include "./kernel_errcode.h"
#include "proto/node_def.pb.h"
#include "common/tensor.h"
#include "proto/attr.pb.h"
#include "aicpu_sharder/aicpu_sharder.h"
namespace {
#define SECUREC_MEM_CPY_MAX_LEN (0x7fffffffUL)
}  // namespace
namespace aicpu {
namespace dataset {
uint32_t ReshapeKernel::DoCompute() {
  size_t type_size = GetDataTypeSize(matrix_info_.matrix_type);
  if (type_size < 1) {
    AICPU_LOGE("don't support input tensor types");
    return kAicpuKernelStateFailed;
  }
  auto dstData = reinterpret_cast<void *>(io_addrs_[1]);
  int64_t dstSize = static_cast<int64_t>(input_size_ * type_size);
  auto srcData = reinterpret_cast<void *>(io_addrs_[0]);
  int64_t srcSize = static_cast<int64_t>(input_size_ * type_size);
  AICPU_LOGD("Begin memcpy tensor data len [%lu]", srcSize);
  std::atomic<bool> task_flag(true);
  auto shard = [&task_flag, &dstData, &srcData, dstSize](int64_t start, int64_t limit) {
    char *dst = reinterpret_cast<char *>(dstData) + start;
    const char *src = reinterpret_cast<const char *>(srcData) + start;
    int64_t len = limit - start;
    if (len < 0L) {
      task_flag.store(false);
      AICPU_LOGE("Len is less than zero, len[%lld]", len);
      return;
    }
    int64_t dstMax = dstSize - start > static_cast<int64_t>(SECUREC_MEM_CPY_MAX_LEN)
                       ? static_cast<int64_t>(SECUREC_MEM_CPY_MAX_LEN)
                       : dstSize - start;
    auto mem_ret = memcpy_s(dst, dstMax, src, len);
    if (mem_ret != EOK) {
      task_flag.store(false);
      AICPU_LOGE("Failed to memcpy tensor data, destSize=[%lld], count=[%lld], errorCode=[%d].", dstSize - start, len,
                 mem_ret);
      return;
    }
  };
  int64_t per_unit_size = 1024LL * 1024LL;  // Copying 1MB costs about 1ms
  ParallelFor(srcSize, per_unit_size, shard);
  if (!task_flag.load()) {
    return kAicpuKernelStateFailed;
  }
  return kAicpuKernelStateSucess;
}

uint32_t ReshapeKernel::ParseKernelParam() {
  AICPU_LOGI("aicpu ReshapeKernel");
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
__attribute__((visibility("default"))) uint32_t Reshape(void *param) {
  aicpu::dataset::ReshapeKernel reshapeKernel;
  return reshapeKernel.Compute(param);
}
}
