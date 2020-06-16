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
ShardShuffle::ShardShuffle(uint32_t seed, ShuffleType shuffle_type)
    : shuffle_seed_(seed),
      no_of_samples_(0),
      replacement_(false),
      reshuffle_each_epoch_(true),
      shuffle_type_(shuffle_type) {}

ShardShuffle::ShardShuffle(uint32_t seed, int64_t no_of_samples, bool replacement, bool reshuffle_each_epoch,
                           ShuffleType shuffle_type)
    : shuffle_seed_(seed),
      no_of_samples_(no_of_samples),
      replacement_(replacement),
      reshuffle_each_epoch_(reshuffle_each_epoch),
      shuffle_type_(shuffle_type) {}

int64_t ShardShuffle::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (replacement_) {
    return no_of_samples_ == 0 ? dataset_size : no_of_samples_;
  }
  return dataset_size;
}

MSRStatus ShardShuffle::Execute(ShardTask &tasks) {
  if (tasks.categories < 1) {
    return FAILED;
  }
  if (shuffle_type_ == kShuffleSample) {  // shuffle each sample
    if (tasks.permutation_.empty() == true) {
      tasks.MakePerm();
    }
    if (replacement_ == true) {
      ShardTask new_tasks;
      if (no_of_samples_ == 0) {
        no_of_samples_ = static_cast<int>(tasks.Size());
      }
      if (no_of_samples_ <= 0) {
        MS_LOG(ERROR) << "no_of_samples need to be positive.";
        return FAILED;
      }
      new_tasks.task_list_.reserve(no_of_samples_);
      for (uint32_t i = 0; i < no_of_samples_; ++i) {
        new_tasks.InsertTask(tasks.GetRandomTask());
      }
      std::swap(tasks, new_tasks);
    } else {
      std::shuffle(tasks.permutation_.begin(), tasks.permutation_.end(), std::default_random_engine(shuffle_seed_));
    }
  } else {  // shuffle unit like: (a1, b1, c1),(a2, b2, c2),..., (an, bn, cn)
    uint32_t individual_size = tasks.Size() / tasks.categories;
    std::vector<std::vector<int>> new_permutations(tasks.categories, std::vector<int>(individual_size));
    for (uint32_t i = 0; i < tasks.categories; i++) {
      for (uint32_t j = 0; j < individual_size; j++) new_permutations[i][j] = static_cast<int>(j);
      std::shuffle(new_permutations[i].begin(), new_permutations[i].end(), std::default_random_engine(shuffle_seed_));
    }
    tasks.permutation_.clear();
    for (uint32_t j = 0; j < individual_size; j++) {
      for (uint32_t i = 0; i < tasks.categories; i++) {
        tasks.permutation_.push_back(new_permutations[i][j] * static_cast<int>(tasks.categories) + static_cast<int>(i));
      }
    }
  }
  if (reshuffle_each_epoch_) shuffle_seed_++;
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
