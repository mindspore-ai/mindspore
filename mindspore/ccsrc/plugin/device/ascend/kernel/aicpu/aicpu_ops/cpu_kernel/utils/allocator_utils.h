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
#ifndef AICPU_UTILS_ALLOCATOR_UTILS_H_
#define AICPU_UTILS_ALLOCATOR_UTILS_H_
#include <functional>
#include <memory>
#include <vector>

#include "cpu_attr_value.h"
#include "cpu_context.h"
#include "context/inc/cpu_node_def.h"
#include "cpu_tensor.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernelAllocatorUtils {
 public:
  static uint32_t ParamCheck(CpuKernelContext &ctx, const std::vector<int64_t> &dims, const void *data_ptr,
                             Tensor *&outputResultTensor);
  static uint32_t UpdateOutputDataTensor(CpuKernelContext &ctx, const std::vector<int64_t> &dims, DataType type,
                                         const void *data_ptr, int64_t input_data_size, Tensor *&outputResultTensor);
  static uint32_t CheckOutputDataPtr(CpuKernelContext &ctx, const uint64_t data_ptr);
  static uint32_t DeleteOutputDataPtr(CpuKernelContext &ctx, const uint64_t data_ptr);
  static int64_t GetInputDataSize(CpuKernelContext &ctx, const std::vector<int64_t> &dims, DataType type);
  static uint32_t AllocateOutputTensorDataMemory(CpuKernelContext &ctx, const std::vector<uint64_t> &shape,
                                                 DataType type, Tensor *&outputResultTensor);
};
}  // namespace aicpu
#endif  // AICPU_UTILS_ALLOCATOR_UTILS_H_
