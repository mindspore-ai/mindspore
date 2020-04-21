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

#ifndef MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_
#define MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "mindrecord/include/shard_operator.h"
#include "mindrecord/include/shard_shuffle.h"
#include "mindrecord/include/shard_category.h"

namespace mindspore {
namespace mindrecord {
class ShardPkSample : public ShardCategory {
 public:
  ShardPkSample(const std::string &category_field, int64_t num_elements);

  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories);

  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories, uint32_t seed);

  ~ShardPkSample() override{};

  MSRStatus suf_execute(ShardTask &tasks) override;

 private:
  bool shuffle_;
  std::shared_ptr<ShardShuffle> shuffle_op_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_
