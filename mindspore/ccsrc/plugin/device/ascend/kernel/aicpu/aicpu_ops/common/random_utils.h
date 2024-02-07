/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef AI_CPU_COMMON_RANDOM_UTILS_H_
#define AI_CPU_COMMON_RANDOM_UTILS_H_
#include <vector>
#include <string>
#include <cstdint>

namespace aicpu {
namespace random {
uint64_t GetRNG(int64_t *seed, int64_t *seed2);
// Get random generator seed for random ops
uint64_t GetSeed(const uint64_t &global_seed, const uint64_t &ops_seed);

// Get random generator for random ops which bases KernelBase
uint64_t GetKernelBaseRandomStates(const std::vector<uintptr_t> &ioAddrs, const uint32_t &counts_index,
                                   const uint32_t &states_index, const uint64_t &seed, const uint64_t &seed2,
                                   const std::string &kernel_name, uint32_t *kernel_ret);
uint64_t InferOutputShape(std::vector<uint64_t> *out_shape, std::vector<uint64_t> *shape,
                          std::vector<uint64_t> *mean_shape, std::vector<uint64_t> *stddev_shape, uint64_t *count);
void NormalizeShape(std::vector<uint64_t> *shape, uint64_t size);
}  // namespace random
}  // namespace aicpu

#endif
