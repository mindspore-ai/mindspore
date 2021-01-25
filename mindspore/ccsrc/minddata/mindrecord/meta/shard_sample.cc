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

#include "minddata/mindrecord/include/shard_sample.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
ShardSample::ShardSample(int n)
    : numerator_(0),
      denominator_(0),
      partition_id_(0),
      no_of_samples_(n),
      indices_({}),
      sampler_type_(kCustomTopNSampler),
      offset_(-1) {}

ShardSample::ShardSample(int num, int den)
    : numerator_(num),
      denominator_(den),
      partition_id_(0),
      no_of_samples_(0),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler),
      offset_(-1) {}

ShardSample::ShardSample(int num, int den, int par, int no_of_samples, int offset)
    : numerator_(num),
      denominator_(den),
      partition_id_(par),
      no_of_samples_(no_of_samples),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler),
      offset_(offset) {}

ShardSample::ShardSample(const std::vector<int64_t> &indices)
    : numerator_(0),
      denominator_(0),
      partition_id_(0),
      no_of_samples_(0),
      indices_(indices),
      sampler_type_(kSubsetSampler) {}

ShardSample::ShardSample(const std::vector<int64_t> &indices, uint32_t seed) : ShardSample(indices) {
  sampler_type_ = kSubsetRandomSampler;
  shuffle_op_ = std::make_shared<ShardShuffle>(seed);
}

int64_t ShardSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (sampler_type_ == kCustomTopNSampler) {
    return no_of_samples_;
  }

  if (sampler_type_ == kCustomTopPercentSampler) {
    if (dataset_size % denominator_ == 0) {
      return dataset_size / denominator_ * numerator_;
    } else {
      return dataset_size / denominator_ * numerator_ + 1;
    }
  }
  if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
    return indices_.size();
  }
  return 0;
}

MSRStatus ShardSample::UpdateTasks(ShardTask &tasks, int taking) {
  if (tasks.permutation_.empty()) {
    ShardTask new_tasks;
    int total_no = static_cast<int>(tasks.Size());
    if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
      for (int i = 0; i < indices_.size(); ++i) {
        int index = ((indices_[i] % total_no) + total_no) % total_no;
        new_tasks.InsertTask(tasks.GetTaskByID(index));  // different mod result between c and python
      }
    } else {
      int count = 0;
      if (nums_per_shard_.empty()) {
        for (size_t i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
          if (no_of_samples_ != 0 && count == no_of_samples_) break;
          new_tasks.InsertTask(tasks.GetTaskByID(i % total_no));  // rounding up. if overflow, go back to start
          count++;
        }
      } else {
        // Get samples within a specific range
        size_t i = partition_id_ - 1 >= 0 ? nums_per_shard_[partition_id_ - 1] : 0;
        for (; i < nums_per_shard_[partition_id_]; i++) {
          if (no_of_samples_ != 0 && count == no_of_samples_) break;
          new_tasks.InsertTask(tasks.GetTaskByID(i % total_no));
          count++;
        }
      }
    }
    std::swap(tasks, new_tasks);
  } else {
    ShardTask new_tasks;
    if (taking > static_cast<int>(tasks.permutation_.size())) {
      return FAILED;
    }
    int total_no = static_cast<int>(tasks.permutation_.size());
    int count = 0;
    for (size_t i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
      if (no_of_samples_ != 0 && count == no_of_samples_) break;
      new_tasks.InsertTask(tasks.GetTaskByID(tasks.permutation_[i % total_no]));
      count++;
    }
    std::swap(tasks, new_tasks);
  }
  return SUCCESS;
}

MSRStatus ShardSample::Execute(ShardTask &tasks) {
  if (offset_ != -1) {
    int64_t old_v = 0;
    int num_rows_ = static_cast<int>(tasks.Size());
    for (int x = 0; x < denominator_; x++) {
      int samples_per_buffer_ = (num_rows_ + offset_) / denominator_;
      int remainder = (num_rows_ + offset_) % denominator_;
      if (x < remainder) samples_per_buffer_++;
      if (x < offset_) samples_per_buffer_--;
      old_v += samples_per_buffer_;
      // nums_per_shard_ is used to save the current shard's ending index
      nums_per_shard_.push_back(old_v);
    }
  }
  int no_of_categories = static_cast<int>(tasks.categories);
  int total_no = static_cast<int>(tasks.Size());  // make sure task_size

  int taking = 0;
  if (sampler_type_ == kCustomTopNSampler) {  // non sharding case constructor #1
    no_of_samples_ = std::min(no_of_samples_, total_no);
    taking = no_of_samples_ - no_of_samples_ % no_of_categories;
  } else if (sampler_type_ == kSubsetRandomSampler || sampler_type_ == kSubsetSampler) {
    if (indices_.size() > total_no) {
      MS_LOG(ERROR) << "parameter indices's size is greater than dataset size.";
      return FAILED;
    }
  } else {  // constructor TopPercent
    if (numerator_ > 0 && denominator_ > 0 && numerator_ <= denominator_) {
      if (numerator_ == 1 && denominator_ > 1) {  // sharding
        taking = (total_no + denominator_ - 1) / denominator_;
      } else {  // non sharding
        taking = total_no * numerator_ / denominator_;
        taking -= (taking % no_of_categories);
      }
    } else {
      MS_LOG(ERROR) << "parameter numerator or denominator is illegal";
      return FAILED;
    }
  }
  return UpdateTasks(tasks, taking);
}

MSRStatus ShardSample::SufExecute(ShardTask &tasks) {
  if (sampler_type_ == kSubsetRandomSampler) {
    if (SUCCESS != (*shuffle_op_)(tasks)) {
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
