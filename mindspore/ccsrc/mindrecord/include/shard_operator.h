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

#ifndef MINDRECORD_INCLUDE_SHARD_OPERATOR_H_
#define MINDRECORD_INCLUDE_SHARD_OPERATOR_H_

#include "mindrecord/include/shard_task.h"

namespace mindspore {
namespace mindrecord {
class ShardOperator {
 public:
  virtual ~ShardOperator() = default;

  MSRStatus operator()(ShardTask &tasks) {
    if (SUCCESS != this->PreExecute(tasks)) {
      return FAILED;
    }
    if (SUCCESS != this->Execute(tasks)) {
      return FAILED;
    }
    if (SUCCESS != this->SufExecute(tasks)) {
      return FAILED;
    }
    return SUCCESS;
  }

  virtual MSRStatus PreExecute(ShardTask &tasks) { return SUCCESS; }

  virtual MSRStatus Execute(ShardTask &tasks) = 0;

  virtual MSRStatus SufExecute(ShardTask &tasks) { return SUCCESS; }

  virtual int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) { return -1; }
};
}  // namespace mindrecord
}  // namespace mindspore
#endif  // MINDRECORD_INCLUDE_SHARD_OPERATOR_H_
