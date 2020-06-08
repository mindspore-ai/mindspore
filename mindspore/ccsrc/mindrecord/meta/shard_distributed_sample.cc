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

#include "mindrecord/include/shard_distributed_sample.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
ShardDistributedSample::ShardDistributedSample(int num_shards, int shard_id, int no_of_padded_samples, bool shuffle,
                                               uint32_t seed)
    : ShardSample(1, num_shards, shard_id),
      shuffle_(shuffle),
      no_of_padded_samples_(no_of_padded_samples),
      init_judgment_(false) {
  shuffle_op_ = std::make_shared<ShardShuffle>(seed, kShuffleSample);
}

ShardDistributedSample::ShardDistributedSample(int num_shards, int shard_id, bool shuffle, uint32_t seed)
    : ShardDistributedSample(num_shards, shard_id, 0, shuffle, seed) {}

int64_t ShardDistributedSample::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (no_of_padded_samples_ <= 0) {
    if (dataset_size % denominator_ == 0) {
      return dataset_size / denominator_ * numerator_;
    } else {
      return dataset_size / denominator_ * numerator_ + 1;
    }
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

MSRStatus ShardDistributedSample::PreExecute(ShardTask &tasks) {
  auto total_no = tasks.Size();
  if (no_of_padded_samples_ > 0 && init_judgment_ == false) {  // we only judge this in first time
    init_judgment_ = true;
    if (total_no % denominator_ != 0) {
      MS_LOG(ERROR) << "Dataset size plus number of padded samples is not divisible by number of shards. "
                    << "task size: " << total_no << ", number padded: " << no_of_padded_samples_
                    << ", denominator: " << denominator_;
      return FAILED;
    }
  }
  if (shuffle_ == true) {
    if (SUCCESS != (*shuffle_op_)(tasks)) {
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
