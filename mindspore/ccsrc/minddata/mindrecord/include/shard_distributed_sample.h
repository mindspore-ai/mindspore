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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_DISTRIBUTED_SAMPLE_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_DISTRIBUTED_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "minddata/mindrecord/include/shard_sample.h"

namespace mindspore {
namespace mindrecord {
class MINDRECORD_API ShardDistributedSample : public ShardSample {
 public:
  ShardDistributedSample(int num_shards, int shard_id, int64_t no_of_padded_samples, bool shuffle, uint32_t seed,
                         int64_t no_of_samples = 0, int64_t offset = -1);

  ShardDistributedSample(int num_shards, int shard_id, bool shuffle, uint32_t seed, int64_t no_of_samples = 0,
                         int64_t offset = -1);

  void SetNumPaddedSamples(int64_t no_of_padded_samples) { no_of_padded_samples_ = no_of_padded_samples; }

  ~ShardDistributedSample() override{};

  Status PreExecute(ShardTaskList &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  bool shuffle_;
  int64_t no_of_padded_samples_;
  bool first_epoch_;    // check (num_sample + num_padded) % num_shards == 0 in first epoch
  ShardTaskList task_;  // maintain the input tasks in first epoch
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_DISTRIBUTED_SAMPLE_H_
