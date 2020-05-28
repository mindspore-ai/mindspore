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

#ifndef MINDRECORD_INCLUDE_SHARD_SHUFFLE_H_
#define MINDRECORD_INCLUDE_SHARD_SHUFFLE_H_

#include <random>
#include "mindrecord/include/shard_operator.h"

namespace mindspore {
namespace mindrecord {
class ShardShuffle : public ShardOperator {
 public:
  explicit ShardShuffle(uint32_t seed = 0, ShuffleType shuffle_type = kShuffleCategory);

  ~ShardShuffle() override{};

  MSRStatus Execute(ShardTask &tasks) override;

 private:
  uint32_t shuffle_seed_;
  ShuffleType shuffle_type_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_SHUFFLE_H_
