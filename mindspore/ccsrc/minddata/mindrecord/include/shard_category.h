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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_CATEGORY_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_CATEGORY_H_

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_operator.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardCategory : public ShardOperator {
 public:
  explicit ShardCategory(const std::vector<std::pair<std::string, std::string>> &categories,
                         int64_t num_elements = std::numeric_limits<int64_t>::max(), bool replacement = false);

  ShardCategory(const std::string &category_field, int64_t num_elements,
                int64_t num_categories = std::numeric_limits<int64_t>::max(), bool replacement = false);

  ~ShardCategory() override{};

  const std::vector<std::pair<std::string, std::string>> &GetCategories() const { return categories_; }

  const std::string GetCategoryField() const { return category_field_; }

  int64_t GetNumElements() const { return num_elements_; }

  int64_t GetNumCategories() const { return num_categories_; }

  bool GetReplacement() const { return replacement_; }

  Status Execute(ShardTaskList &tasks) override;

  int64_t GetNumSamples(int64_t dataset_size, int64_t num_classes) override;

 private:
  std::vector<std::pair<std::string, std::string>> categories_;
  std::string category_field_;
  int64_t num_elements_;
  int64_t num_categories_;
  bool replacement_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_CATEGORY_H_
