/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "common/random_utils.h"
#include <securec.h>
#include <random>
#include <string>
#include <vector>
#include "utils/philox_random.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
namespace random {
uint64_t GetSeed(const uint64_t &global_seed, const uint64_t &ops_seed) {
  uint64_t seed = 0;
  if (global_seed == 0 && ops_seed == 0) {
    std::random_device r;
    seed = static_cast<uint64_t>(r());
  } else {
    // Using Philox algorithm to scramble the global_seed and ops_seed so that
    // the user doesn't need to worry about which seed is more important.
    random::PhiloxRandom::Key outer_key;
    random::PhiloxRandom::ResultType outer_counter;
    outer_key[0] = 0x3ec8f720;
    outer_key[1] = 0x02461e29;
    outer_counter[0] = static_cast<uint32_t>(global_seed);
    outer_counter[1] = static_cast<uint32_t>(global_seed >> 32);
    outer_counter[2] = static_cast<uint32_t>(ops_seed);
    outer_counter[3] = static_cast<uint32_t>(ops_seed >> 32);
    const auto seed_mix = random::PhiloxRandom(outer_counter, outer_key)();
    uint64_t seed_low = seed_mix[0];
    uint64_t seed_high = seed_mix[1];
    seed = (seed_high << 32) | seed_low;
  }
  return seed;
}

uint64_t GetKernelBaseRandomStates(const std::vector<uintptr_t> &ioAddrs, const uint32_t &counts_index,
                                   const uint32_t &states_index, const uint64_t &seed, const uint64_t &seed2,
                                   const std::string &kernel_name, uint32_t *kernel_ret) {
  uint64_t *counts_ptr = reinterpret_cast<uint64_t *>(ioAddrs[counts_index]);
  uint64_t *states_ptr = reinterpret_cast<uint64_t *>(ioAddrs[states_index]);

  if (counts_ptr == nullptr) {
    AICPU_LOGE("For AICPU ops ", kernel_name, ", the points of counts is a nullptr, which is invalid!");
    *kernel_ret = kAicpuKernelStateFailed;
    return 0;
  }
  if (states_ptr == nullptr) {
    AICPU_LOGE("For AICPU ops ", kernel_name, ", the points of states is a nullptr, which is invalid!");
    *kernel_ret = kAicpuKernelStateFailed;
    return 0;
  }

  // count the execution times of op
  uint64_t counts = *counts_ptr;
  // seed of the op, passed between executions, which make op stateful
  uint64_t states = *states_ptr;

  // setup seed
  std::mt19937 seed_rng;
  uint64_t final_seed = 0;
  if (counts == 0) {
    counts_ptr[0] = 1;
    final_seed = GetSeed(seed, seed2);
  } else {
    final_seed = states;
  }
  seed_rng.seed(final_seed);

  // update random state
  states_ptr[0] = static_cast<uint64_t>(seed_rng());

  return final_seed;
}
}  // namespace random
}  // namespace aicpu
