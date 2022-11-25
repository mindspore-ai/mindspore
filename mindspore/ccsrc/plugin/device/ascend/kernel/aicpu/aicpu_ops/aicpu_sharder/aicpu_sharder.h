/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef AICPU_OPS_AICPU_SHARDER_H_
#define AICPU_OPS_AICPU_SHARDER_H_

#include <functional>
#include "common/kernel_util.h"

namespace aicpu {
using SharderWork = std::function<void(int64_t, int64_t)>;
}  // namespace aicpu

extern "C" {
/**
 * Shards the "total" unit of work refer "perUintSize"
 * @param total Total unit of work
 * @param per_unit_size Minimum shard unit
 * @param work should be a callable taking (int64, int64) arguments.
                 work(start, limit) computes the work units from [start, limit),
                i.e., [start, limit) is a shard.
 */
__attribute__((weak)) AICPU_VISIBILITY_API void ParallelFor(int64_t total, int64_t per_unit_size,
                                                            const aicpu::SharderWork &work);

/**
 * Get CPU number
 * @param None
 * @return CPU number
 */
__attribute__((weak)) AICPU_VISIBILITY_API uint32_t GetCPUNum();
}

#endif  // AICPU_OPS_AICPU_SHARDER_H_
