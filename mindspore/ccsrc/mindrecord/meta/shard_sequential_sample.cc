/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "mindrecord/include/shard_sequential_sample.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
ShardSequentialSample::ShardSequentialSample(int n, int offset)
    : ShardSample(n), offset_(offset), per_(0.0f), per_offset_(0.0f) {}

ShardSequentialSample::ShardSequentialSample(float per, float per_offset)
    : ShardSample(0), offset_(0), per_(per), per_offset_(per_offset) {}

int64_t ShardSequentialSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (no_of_samples_ == 0 && (per_ >= -kEpsilon && per_ <= kEpsilon)) {
    return dataset_size;
  }
  if (per_ > kEpsilon && per_ <= 1.0f) {
    return dataset_size * kEpsilon;
  }
  return no_of_samples_;
}

MSRStatus ShardSequentialSample::Execute(ShardTask &tasks) {
  int total_no = static_cast<int>(tasks.Size());
  int taking;
  if (no_of_samples_ == 0 && (per_ >= -kEpsilon && per_ <= kEpsilon)) {
    taking = total_no;
  } else if (per_ > kEpsilon && per_ <= 1.0f) {
    taking = total_no * kEpsilon;
  } else {
    taking = no_of_samples_;
  }

  if (tasks.permutation_.empty()) {
    ShardTask new_tasks;
    total_no = static_cast<int>(tasks.Size());
    for (int i = offset_; i < taking + offset_; ++i) {
      new_tasks.InsertTask(tasks.GetTaskByID(i % total_no));
    }
    std::swap(tasks, new_tasks);
  } else {  // shuffled
    ShardTask new_tasks;
    if (taking > static_cast<int>(tasks.permutation_.size())) {
      return FAILED;
    }
    total_no = static_cast<int>(tasks.permutation_.size());
    for (size_t i = offset_; i < taking + offset_; ++i) {
      new_tasks.InsertTask(tasks.GetTaskByID(tasks.permutation_[i % total_no]));
    }
    std::swap(tasks, new_tasks);
  }
  return SUCCESS;
}

}  // namespace mindrecord
}  // namespace mindspore
