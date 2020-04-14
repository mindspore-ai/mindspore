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

#include "mindrecord/include/shard_shuffle.h"

#include <algorithm>

namespace mindspore {
namespace mindrecord {
ShardShuffle::ShardShuffle(uint32_t seed) : shuffle_seed_(seed) {}

MSRStatus ShardShuffle::execute(ShardTask &tasks) {
  if (tasks.categories < 1) {
    return FAILED;
  }
  uint32_t individual_size = tasks.Size() / tasks.categories;
  std::vector<std::vector<int>> new_permutations(tasks.categories, std::vector<int>(individual_size));
  for (uint32_t i = 0; i < tasks.categories; i++) {
    for (uint32_t j = 0; j < individual_size; j++) new_permutations[i][j] = static_cast<int>(j);
    std::shuffle(new_permutations[i].begin(), new_permutations[i].end(), std::default_random_engine(shuffle_seed_));
  }
  shuffle_seed_++;
  tasks.permutation_.clear();
  for (uint32_t j = 0; j < individual_size; j++) {
    for (uint32_t i = 0; i < tasks.categories; i++) {
      tasks.permutation_.push_back(new_permutations[i][j] * static_cast<int>(tasks.categories) + static_cast<int>(i));
    }
  }
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
