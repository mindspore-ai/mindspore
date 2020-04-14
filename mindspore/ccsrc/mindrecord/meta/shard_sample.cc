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

#include "mindrecord/include/shard_sample.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
ShardSample::ShardSample(int n)
    : numerator_(0),
      denominator_(0),
      no_of_samples_(n),
      partition_id_(0),
      indices_({}),
      sampler_type_(kCustomTopNSampler) {}

ShardSample::ShardSample(int num, int den)
    : numerator_(num),
      denominator_(den),
      no_of_samples_(0),
      partition_id_(0),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler) {}

ShardSample::ShardSample(int num, int den, int par)
    : numerator_(num),
      denominator_(den),
      no_of_samples_(0),
      partition_id_(par),
      indices_({}),
      sampler_type_(kCustomTopPercentSampler) {}

ShardSample::ShardSample(const std::vector<int64_t> &indices, uint32_t seed)
    : numerator_(0),
      denominator_(0),
      no_of_samples_(0),
      partition_id_(0),
      indices_(indices),
      sampler_type_(kSubsetRandomSampler) {
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
  if (sampler_type_ == kSubsetRandomSampler) {
    return indices_.size();
  }
  return -1;
}

const std::pair<int, int> ShardSample::get_partitions() const {
  if (numerator_ == 1 && denominator_ > 1) {
    return std::pair<int, int>(denominator_, partition_id_);
  }
  return std::pair<int, int>(-1, -1);
}

MSRStatus ShardSample::execute(ShardTask &tasks) {
  int no_of_categories = static_cast<int>(tasks.categories);
  int total_no = static_cast<int>(tasks.Size());

  int taking = 0;
  if (sampler_type_ == kCustomTopNSampler) {  // non sharding case constructor #1
    no_of_samples_ = std::min(no_of_samples_, total_no);
    taking = no_of_samples_ - no_of_samples_ % no_of_categories;
  } else if (sampler_type_ == kSubsetRandomSampler) {
    if (indices_.size() > total_no) {
      MS_LOG(ERROR) << "parameter indices's size is greater than dataset size.";
      return FAILED;
    }
  } else {  // constructor TopPercent
    if (numerator_ > 0 && denominator_ > 0 && numerator_ <= denominator_) {
      if (numerator_ == 1 && denominator_ > 1) {  // sharding
        taking = (total_no / denominator_) + (total_no % denominator_ == 0 ? 0 : 1);
      } else {  // non sharding
        taking = total_no * numerator_ / denominator_;
        taking -= (taking % no_of_categories);
      }
    } else {
      MS_LOG(ERROR) << "parameter numerator or denominator is illegal";
      return FAILED;
    }
  }

  if (tasks.permutation_.empty()) {
    ShardTask new_tasks;
    total_no = static_cast<int>(tasks.Size());
    if (sampler_type_ == kSubsetRandomSampler) {
      for (int i = 0; i < indices_.size(); ++i) {
        int index = ((indices_[i] % total_no) + total_no) % total_no;
        new_tasks.InsertTask(tasks.get_task_by_id(index));  // different mod result between c and python
      }
    } else {
      for (int i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
        new_tasks.InsertTask(tasks.get_task_by_id(i % total_no));  // rounding up. if overflow, go back to start
      }
    }
    std::swap(tasks, new_tasks);
  } else {
    ShardTask new_tasks;
    if (taking > static_cast<int>(tasks.permutation_.size())) {
      return FAILED;
    }
    total_no = static_cast<int>(tasks.permutation_.size());
    for (size_t i = partition_id_ * taking; i < (partition_id_ + 1) * taking; i++) {
      new_tasks.InsertTask(tasks.get_task_by_id(tasks.permutation_[i % total_no]));
    }
    std::swap(tasks, new_tasks);
  }
  return SUCCESS;
}

MSRStatus ShardSample::suf_execute(ShardTask &tasks) {
  if (sampler_type_ == kSubsetRandomSampler) {
    if (SUCCESS != (*shuffle_op_)(tasks)) {
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
