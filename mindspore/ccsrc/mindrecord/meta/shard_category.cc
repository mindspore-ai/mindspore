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

#include "mindrecord/include/shard_category.h"

namespace mindspore {
namespace mindrecord {
ShardCategory::ShardCategory(const std::vector<std::pair<std::string, std::string>> &categories)
    : categories_(categories) {}

const std::vector<std::pair<std::string, std::string>> &ShardCategory::get_categories() const { return categories_; }

MSRStatus ShardCategory::operator()(ShardTask &tasks) { return SUCCESS; }
}  // namespace mindrecord
}  // namespace mindspore
