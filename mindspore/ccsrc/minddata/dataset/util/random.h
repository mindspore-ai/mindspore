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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_RANDOM_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_RANDOM_H_

#if defined(_WIN32) || defined(_WIN64)
#ifndef _CRT_RAND_S
#define _CRT_RAND_S
#endif
#include <stdlib.h>
#endif
#include <chrono>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
inline std::mt19937 GetRandomDevice() {
#if defined(_WIN32) || defined(_WIN64)
  unsigned int number;
  rand_s(&number);
  std::mt19937 random_device{static_cast<uint32_t>(number)};
#else
  int i = 0;
  constexpr int64_t retry_times = 5;
  while (i < retry_times) {
    try {
      std::mt19937 random_device{std::random_device("/dev/urandom")()};
      return random_device;
    } catch (const std::exception &e) {
      MS_LOG(WARNING) << "Get std::random_device failed, retry: " << i << ", error: " << e.what();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      i++;
    }
  }
  std::mt19937 random_device{std::random_device("/dev/urandom")()};
#endif
  return random_device;
}

inline uint32_t GetNewSeed() {
  std::mt19937 random_device = GetRandomDevice();
  std::uniform_int_distribution<uint32_t> distribution(0, std::numeric_limits<uint32_t>::max());
  return distribution(random_device);
}

inline uint32_t GetSeed() {
  uint32_t seed = GlobalContext::config_manager()->seed();
  if (seed == std::mt19937::default_seed) {
    seed = GetNewSeed();
  }
  return seed;
}

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_RANDOM_H_
