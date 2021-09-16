/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_shuffle.h"

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
  return no_of_samples_ == 0 ? dataset_size : std::min(dataset_size, no_of_samples_);
}

Status ShardShuffle::CategoryShuffle(ShardTaskList &tasks) {
  uint32_t individual_size = tasks.sample_ids_.size() / tasks.categories;
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

  ShardTaskList new_tasks;
  for (size_t i = 0; i < individual_size; ++i) {
    new_tasks.AssignTask(tasks, tasks.permutation_[i]);
  }
  ShardTaskList::TaskListSwap(tasks, new_tasks);

  return Status::OK();
}

Status ShardShuffle::ShuffleFiles(ShardTaskList &tasks) {
  if (no_of_samples_ == 0) {
    no_of_samples_ = static_cast<int>(tasks.Size());
  }
  CHECK_FAIL_RETURN_UNEXPECTED(no_of_samples_ > 0, "Invalid input, Number of samples [" +
                                                     std::to_string(no_of_samples_) + "] need to be positive.");
  auto shard_sample_cout = GetShardSampleCount();

  // shuffle the files index
  std::vector<uint32_t> shuffle_files;
  for (uint32_t i = 0; i < shard_sample_cout.size(); i++) {
    shuffle_files.push_back(i);
  }
  std::shuffle(shuffle_files.begin(), shuffle_files.end(), std::default_random_engine(shuffle_seed_));

  // reconstruct the permutation between files
  // -- before --
  // file1: [0, 1, 2]
  // file2: [3, 4, 5, 6]
  // file3: [7, 8]
  // file4: [9, 10]
  // files: [file1, file2, file3, file4]
  // permutation: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  // -- after --
  // files: [file4, file1, file3, file2]
  // permutation : [9, 10, 0, 1, 2, 7, 8, 3, 4, 5, 6]
  auto original_permutation = tasks.permutation_;
  uint32_t whole_index = 0;
  for (uint32_t i = 0; i < shuffle_files.size(); i++) {
    uint32_t start_index = 0;
    uint32_t current_size = 0;
    if (shuffle_files[i] == 0) {
      start_index = 0;
      current_size = shard_sample_cout[shuffle_files[i]];
    } else {
      start_index = shard_sample_cout[shuffle_files[i] - 1];
      current_size = shard_sample_cout[shuffle_files[i]] - start_index;
    }
    std::copy(original_permutation.begin() + start_index, original_permutation.begin() + start_index + current_size,
              tasks.permutation_.begin() + whole_index);
    whole_index += current_size;
  }

  auto total_no = static_cast<int64_t>(tasks.Size());
  size_t samples_to_assign =
    (no_of_samples_ > 0 && no_of_samples_ < total_no) ? no_of_samples_ : tasks.sample_ids_.size();
  ShardTaskList new_tasks;
  for (size_t i = 0; i < samples_to_assign; ++i) {
    new_tasks.AssignTask(tasks, tasks.permutation_[i]);
  }
  ShardTaskList::TaskListSwap(tasks, new_tasks);
  return Status::OK();
}

Status ShardShuffle::ShuffleInfile(ShardTaskList &tasks) {
  if (no_of_samples_ == 0) {
    no_of_samples_ = static_cast<int>(tasks.Size());
  }
  CHECK_FAIL_RETURN_UNEXPECTED(no_of_samples_ > 0, "Invalid input, Number of samples [" +
                                                     std::to_string(no_of_samples_) + "] need to be positive.");
  // reconstruct the permutation in file
  // -- before --
  // file1: [0, 1, 2]
  // file2: [3, 4, 5, 6]
  // file3: [7, 8]
  // file4: [9, 10]
  // files: [file1, file2, file3, file4]
  // permutation: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  // -- after --
  // permutation: [2, 0, 1, 4, 6, 3, 5, 8, 7, 9, 10]
  auto shard_sample_cout = GetShardSampleCount();
  uint32_t start_index = 0;
  for (uint32_t i = 0; i < shard_sample_cout.size(); i++) {
    auto current_size = shard_sample_cout[i] - start_index;
    std::shuffle(tasks.permutation_.begin() + start_index, tasks.permutation_.begin() + start_index + current_size,
                 std::default_random_engine(shuffle_seed_));
    start_index = shard_sample_cout[i];
  }
  auto total_no = static_cast<int64_t>(tasks.Size());
  ShardTaskList new_tasks;
  size_t samples_to_assign =
    (no_of_samples_ > 0 && no_of_samples_ < total_no) ? no_of_samples_ : tasks.sample_ids_.size();
  for (size_t i = 0; i < samples_to_assign; ++i) {
    new_tasks.AssignTask(tasks, tasks.permutation_[i]);
  }
  ShardTaskList::TaskListSwap(tasks, new_tasks);
  return Status::OK();
}

Status ShardShuffle::Execute(ShardTaskList &tasks) {
  if (reshuffle_each_epoch_) {
    shuffle_seed_++;
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    tasks.categories >= 1,
    "Invalid data, task categories [" + std::to_string(tasks.categories) + "] need to be larger than 1.");
  if (shuffle_type_ == kShuffleSample) {  // shuffle each sample
    if (tasks.permutation_.empty() == true) {
      tasks.MakePerm();
    }
    if (GetShuffleMode() == dataset::ShuffleMode::kGlobal) {
      if (replacement_ == true) {
        ShardTaskList new_tasks;
        if (no_of_samples_ == 0) no_of_samples_ = static_cast<int>(tasks.sample_ids_.size());
        CHECK_FAIL_RETURN_UNEXPECTED(no_of_samples_ > 0, "Invalid input, Number of samples [" +
                                                           std::to_string(no_of_samples_) + "] need to be positive.");
        for (uint32_t i = 0; i < no_of_samples_; ++i) {
          new_tasks.AssignTask(tasks, tasks.GetRandomTaskID());
        }

        ShardTaskList::TaskListSwap(tasks, new_tasks);
      } else {
        std::shuffle(tasks.permutation_.begin(), tasks.permutation_.end(), std::default_random_engine(shuffle_seed_));
        auto total_no = static_cast<int64_t>(tasks.Size());
        ShardTaskList new_tasks;
        size_t samples_to_assign =
          (no_of_samples_ > 0 && no_of_samples_ < total_no) ? no_of_samples_ : tasks.sample_ids_.size();
        for (size_t i = 0; i < samples_to_assign; ++i) {
          new_tasks.AssignTask(tasks, tasks.permutation_[i]);
        }
        ShardTaskList::TaskListSwap(tasks, new_tasks);
      }
    } else if (GetShuffleMode() == dataset::ShuffleMode::kInfile) {
      RETURN_IF_NOT_OK(ShuffleInfile(tasks));
    } else if (GetShuffleMode() == dataset::ShuffleMode::kFiles) {
      RETURN_IF_NOT_OK(ShuffleFiles(tasks));
    }
  } else {  // shuffle unit like: (a1, b1, c1),(a2, b2, c2),..., (an, bn, cn)
    return this->CategoryShuffle(tasks);
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
