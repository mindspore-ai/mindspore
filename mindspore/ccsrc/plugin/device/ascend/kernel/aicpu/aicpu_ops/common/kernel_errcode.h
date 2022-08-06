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
#ifndef AICPU_OPS_AICPU_COMMON_KENERL_ERRCODE_H_
#define AICPU_OPS_AICPU_COMMON_KENERL_ERRCODE_H_
#include <cstdint>

namespace aicpu {
constexpr uint32_t kAicpuKernelStateSucess = 0;
constexpr uint32_t kAicpuKernelStateInvalid = 1;
constexpr uint32_t kAicpuKernelStateFailed = 2;
constexpr uint32_t kAicpuKernelStateExecuteTimeout = 3;
constexpr uint32_t kAicpuKernelStateInternalError = 4;
constexpr uint32_t kAicpuKernelStateEndOfSequence = 201;
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_KENERL_ERRCODE_H_
