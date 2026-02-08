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

#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"
#include <random>

namespace mindspore {
namespace kernel {
namespace random {
uint64_t GetSeed(const uint64_t &global_seed, const uint64_t &ops_seed) {
  uint64_t seed = 0;
  if (global_seed == 0 && ops_seed == 0) {
    std::random_device r;
    seed = static_cast<uint64_t>(r());
  } else {
    // Using Philox algorithm to scramble the global_seed and ops_seed so that
    // the user doesn't need to worry about which seed is more important.
    PhiloxRandom::Key outer_key;
    PhiloxRandom::ResultType outer_counter;
    outer_key[0] = 0x3ec8f720;
    outer_key[1] = 0x02461e29;
    outer_counter[0] = static_cast<uint32_t>(global_seed);
    outer_counter[1] = static_cast<uint32_t>(global_seed >> 32);
    outer_counter[2] = static_cast<uint32_t>(ops_seed);
    outer_counter[3] = static_cast<uint32_t>(ops_seed >> 32);
    const auto seed_mix = PhiloxRandom(outer_counter, outer_key)();
    uint64_t seed_low = seed_mix[0];
    uint64_t seed_high = seed_mix[1];
    seed = (seed_high << 32) | seed_low;
  }
  return seed;
}
}  // namespace random
}  // namespace kernel
}  // namespace mindspore
