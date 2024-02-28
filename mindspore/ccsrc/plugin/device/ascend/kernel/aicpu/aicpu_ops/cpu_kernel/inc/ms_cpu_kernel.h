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
#ifndef AICPU_CPU_KERNEL_INC_MS_CPU_KERNEL_H
#define AICPU_CPU_KERNEL_INC_MS_CPU_KERNEL_H
#include "inc/cpu_kernel.h"

#define REGISTER_CUST_KERNEL(type, clazz)                      \
  std::shared_ptr<CpuKernel> Creator_Cust##type##_Kernel() {   \
    std::shared_ptr<clazz> ptr = nullptr;                      \
    ptr = MakeShared<clazz>();                                 \
    return ptr;                                                \
  }                                                            \
  bool g_Cust##type##_Kernel_Creator __attribute__((unused)) = \
    RegistCpuKernel("Cust" + static_cast<std::string>(type), Creator_Cust##type##_Kernel)

#define REGISTER_MS_CPU_KERNEL(type, clazz) \
  REGISTER_CPU_KERNEL(type, clazz);         \
  REGISTER_CUST_KERNEL(type, clazz);

enum KernelStatus : uint32_t {
  KERNEL_STATUS_OK = 0,
  KERNEL_STATUS_PARAM_INVALID = 1,
  KERNEL_STATUS_INNER_ERROR = 2,
  KERNEL_STATUS_TIMEOUT = 3,
  KERNEL_STATUS_PROTOBUF_ERROR = 4,
  KERNEL_STATUS_SHARDER_ERROR = 5,
  KERNEL_STATUS_END_OF_SEQUENCE = 201
};
#endif
