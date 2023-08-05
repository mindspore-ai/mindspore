/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef AI_CPU_RANDOM_UTILS_H_
#define AI_CPU_RANDOM_UTILS_H_
#include <cstdint>
#include <string>
#include "common/kernel_base.h"
#include "cpu_kernel/inc/cpu_ops_kernel.h"

namespace aicpu {
namespace random {
// Return a 64-bit random value.  Different sequences are generated
// in different processes.
uint64_t New64();

// Helper function to convert a 32-bit integer to a float between [0..1).
float Uint32ToFloat(const uint32_t &x);

// Helper function to convert two 32-bit integers to a double between [0..1).
double Uint64ToDouble(const uint32_t &x0, const uint32_t &x1);

// Helper function to convert two 32-bit uniform integers to two floats
// under the unit normal distribution.
void BoxMullerFloat(const uint32_t &x0, const uint32_t &x1, float *f0, float *f1);

// Helper function to convert four 32-bit uniform integers to two doubles
// under the unit normal distribution.
void BoxMullerDouble(const uint32_t &x0, const uint32_t &x1, const uint32_t &x2, const uint32_t &x3, double *d0,
                     double *d1);

// Get random generator seed for random ops
uint64_t GetSeed(const uint64_t &global_seed, const uint64_t &ops_seed);

// Get random generator for random ops which bases CpuKernel
uint64_t GetCpuKernelRandomStates(const CpuKernelContext &ctx, const uint32_t &counts_index,
                                  const uint32_t &states_index, const uint64_t &seed, const uint64_t &seed2,
                                  const std::string &kernel_name, uint32_t *kernel_ret);
}  // namespace random
}  // namespace aicpu

#endif
