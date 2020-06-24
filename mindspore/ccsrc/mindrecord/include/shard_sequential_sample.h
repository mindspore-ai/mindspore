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

#ifndef MINDRECORD_INCLUDE_SHARD_SEQUENTIAL_SAMPLE_H_
#define MINDRECORD_INCLUDE_SHARD_SEQUENTIAL_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "mindrecord/include/shard_sample.h"

namespace mindspore {
namespace mindrecord {
class ShardSequentialSample : public ShardSample {
 public:
  ShardSequentialSample(int n, int offset);

  ShardSequentialSample(float per, float per_offset);

  ~ShardSequentialSample() override{};

  MSRStatus Execute(ShardTask &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  int offset_;
  float per_;
  float per_offset_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_SEQUENTIAL_SAMPLE_H_
