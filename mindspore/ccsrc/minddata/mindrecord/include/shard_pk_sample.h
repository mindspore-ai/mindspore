/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_operator.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "minddata/mindrecord/include/shard_category.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardPkSample : public ShardCategory {
 public:
  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_samples);

  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories, int64_t num_samples);

  ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories, uint32_t seed,
                int64_t num_samples);

  ~ShardPkSample() override{};

  Status SufExecute(ShardTaskList &tasks) override;

  int64_t GetNumSamples() const { return num_samples_; }

 private:
  bool shuffle_;
  std::shared_ptr<ShardShuffle> shuffle_op_;
  int64_t num_samples_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PK_SAMPLE_H_
