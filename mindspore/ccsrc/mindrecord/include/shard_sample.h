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

#ifndef MINDRECORD_INCLUDE_SHARD_SAMPLE_H_
#define MINDRECORD_INCLUDE_SHARD_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "mindrecord/include/shard_operator.h"
#include "mindrecord/include/shard_shuffle.h"

namespace mindspore {
namespace mindrecord {
class ShardSample : public ShardOperator {
 public:
  explicit ShardSample(int n);

  ShardSample(int num, int den);

  ShardSample(int num, int den, int par);

  ShardSample(const std::vector<int64_t> &indices, uint32_t seed);

  ~ShardSample() override{};

  const std::pair<int, int> GetPartitions() const;

  MSRStatus Execute(ShardTask &tasks) override;

  MSRStatus SufExecute(ShardTask &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  int numerator_;
  int denominator_;
  int no_of_samples_;
  int partition_id_;
  std::vector<int64_t> indices_;
  SamplerType sampler_type_;
  std::shared_ptr<ShardShuffle> shuffle_op_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_SAMPLE_H_
