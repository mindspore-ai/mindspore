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

#include "mindrecord/include/shard_pk_sample.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;

namespace mindspore {
namespace mindrecord {
ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements)
    : ShardCategory(category_field, num_elements, std::numeric_limits<int64_t>::max(), true), shuffle_(false) {}

ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories)
    : ShardCategory(category_field, num_elements, num_categories, true), shuffle_(false) {}

ShardPkSample::ShardPkSample(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             uint32_t seed)
    : ShardCategory(category_field, num_elements, num_categories, true), shuffle_(true) {
  shuffle_op_ = std::make_shared<ShardShuffle>(seed, kShuffleSample);  // do shuffle and replacement
}

MSRStatus ShardPkSample::suf_execute(ShardTask &tasks) {
  if (shuffle_ == true) {
    if (SUCCESS != (*shuffle_op_)(tasks)) {
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace mindrecord
}  // namespace mindspore
