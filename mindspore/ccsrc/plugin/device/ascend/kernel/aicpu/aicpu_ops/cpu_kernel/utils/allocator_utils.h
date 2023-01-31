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

#include "cpu_kernel/inc/cpu_attr_value.h"
#include "cpu_kernel/inc/cpu_context.h"
#include "cpu_kernel/common/cpu_node_def.h"
#include "cpu_kernel/inc/cpu_tensor.h"

namespace aicpu {
class AICPU_VISIBILITY CpuKernelAllocatorUtils {
 public:
  static uint32_t ParamCheck(const std::vector<int64_t> &dims, const void *data_ptr, Tensor *&outputResultTensor);
  static uint32_t UpdateOutputDataTensor(const std::vector<int64_t> &dims, DataType type, const void *data_ptr,
                                         int64_t input_data_size, Tensor *&outputResultTensor);
  static uint32_t CheckOutputDataPtr(const uint64_t data_ptr);
  static uint32_t DeleteOutputDataPtr(const uint64_t data_ptr);
  static int64_t GetInputDataSize(const std::vector<int64_t> &dims, DataType type);
};
}  // namespace aicpu
#endif  // AICPU_UTILS_ALLOCATOR_UTILS_H_
