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

#include <utility>
#include "mindrecord/include/shard_operator.h"

namespace mindspore {
namespace mindrecord {
class ShardSample : public ShardOperator {
 public:
  explicit ShardSample(int n);

  ShardSample(int num, int den);

  ShardSample(int num, int den, int par);

  ~ShardSample() override{};

  const std::pair<int, int> get_partitions() const;

  MSRStatus operator()(ShardTask &tasks) override;

 private:
  int numerator_;
  int denominator_;
  int no_of_samples_;
  int partition_id_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_SAMPLE_H_
