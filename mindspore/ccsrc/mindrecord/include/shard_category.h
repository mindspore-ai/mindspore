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

#ifndef MINDRECORD_INCLUDE_SHARD_CATEGORY_H_
#define MINDRECORD_INCLUDE_SHARD_CATEGORY_H_

#include <string>
#include <utility>
#include <vector>
#include "mindrecord/include/shard_operator.h"

namespace mindspore {
namespace mindrecord {
class ShardCategory : public ShardOperator {
 public:
  explicit ShardCategory(const std::vector<std::pair<std::string, std::string>> &categories);

  ~ShardCategory() override{};

  const std::vector<std::pair<std::string, std::string>> &get_categories() const;

  MSRStatus execute(ShardTask &tasks) override;

 private:
  std::vector<std::pair<std::string, std::string>> categories_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDRECORD_INCLUDE_SHARD_CATEGORY_H_
