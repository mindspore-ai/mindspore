/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_distributed_sample.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
ShardDistributedSample::ShardDistributedSample(int num_shards, int shard_id, int no_of_padded_samples, bool shuffle,
                                               uint32_t seed, int no_of_samples, int offset)
    : ShardSample(1, num_shards, shard_id, no_of_samples, offset),
      shuffle_(shuffle),
      no_of_padded_samples_(no_of_padded_samples),
      first_epoch_(true) {
  shuffle_op_ = std::make_shared<ShardShuffle>(seed, kShuffleSample);
}

ShardDistributedSample::ShardDistributedSample(int num_shards, int shard_id, bool shuffle, uint32_t seed,
                                               int no_of_samples, int offset)
    : ShardDistributedSample(num_shards, shard_id, 0, shuffle, seed, no_of_samples, offset) {}

int64_t ShardDistributedSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (no_of_padded_samples_ <= 0) {
    int64_t res = 0;
    if (dataset_size % denominator_ == 0) {
      res = dataset_size / denominator_ * numerator_;
    } else {
      res = dataset_size / denominator_ * numerator_ + 1;
    }
    return no_of_samples_ == 0 ? res : std::min(static_cast<int64_t>(no_of_samples_), res);
  } else {
    auto padded_size = dataset_size + no_of_padded_samples_;
    if (padded_size % denominator_ == 0) {
      return padded_size / denominator_ * numerator_;
    } else {
      return -1;
    }
  }
  return 0;
}

Status ShardDistributedSample::PreExecute(ShardTaskList &tasks) {
  auto total_no = tasks.Size();
  if (no_of_padded_samples_ > 0 && first_epoch_) {
    CHECK_FAIL_RETURN_UNEXPECTED(
      total_no % denominator_ == 0,
      "Invalid input, number of padding samples: " + std::to_string(no_of_padded_samples_) +
        " plus dataset size is not divisible by num_shards: " + std::to_string(denominator_) + ".");
  }
  if (first_epoch_) {
    first_epoch_ = false;
    task_ = tasks;
  } else {
    tasks = task_;
  }
  if (shuffle_ == true) {
    shuffle_op_->SetShardSampleCount(GetShardSampleCount());
    shuffle_op_->UpdateShuffleMode(GetShuffleMode());
    RETURN_IF_NOT_OK((*shuffle_op_)(tasks));
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
