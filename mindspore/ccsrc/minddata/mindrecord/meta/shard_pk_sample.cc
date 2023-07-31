/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_pk_sample.h"

namespace mindspore {
namespace mindrecord {
ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_samples)
    : ShardCategory(category_field, num_elements, std::numeric_limits<int64_t>::max(), true),
      shuffle_(false),
      num_samples_(num_samples) {}

ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             int64_t num_samples)
    : ShardCategory(category_field, num_elements, num_categories, true), shuffle_(false), num_samples_(num_samples) {}

ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             uint32_t seed, int64_t num_samples)
    : ShardCategory(category_field, num_elements, num_categories, true), shuffle_(true), num_samples_(num_samples) {
  shuffle_op_ = std::make_shared<ShardShuffle>(seed, kShuffleSample);  // do shuffle and replacement
}

Status ShardPkSample::SufExecute(ShardTaskList &tasks) {
  if (shuffle_) {
    RETURN_IF_NOT_OK_MR((*shuffle_op_)(tasks));
  }
  return Status::OK();
}
}  // namespace mindrecord
}  // namespace mindspore
