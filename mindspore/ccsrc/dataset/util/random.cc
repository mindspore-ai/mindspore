/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_UTIL_RANDOM_H_
#define DATASET_UTIL_RANDOM_H_

#include "dataset/util/random.h"

#if defined(_WIN32) || defined(_WIn64)
#include <stdlib.h>
#endif
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "dataset/core/config_manager.h"
#include "dataset/core/global_context.h"
#include "dataset/util/status.h"

namespace mindspore {
namespace dataset {
uint32_t GetSeed() {
  uint32_t seed = GlobalContext::config_manager()->seed();
  if (seed == std::mt19937::default_seed) {
#if defined(_WIN32) || defined(_WIN64)
    unsigned int number;
    rand_s(&number);
    std::mt19937 random_device{static_cast<uint32_t>(number)};
#else
    std::random_device random_device("/dev/urandom");
#endif
    std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
    seed = distribution(random_device);
  }

  return seed;
}
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_RANDOM_H_
