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

#include "minddata/mindrecord/include/shard_index.h"

namespace mindspore {
namespace mindrecord {
// table name for index
const char TABLENAME[] = "index_table";

Index::Index() : database_name_(""), table_name_(TABLENAME) {}

void Index::AddIndexField(const int64_t &schemaId, const std::string &field) {
  (void)fields_.emplace_back(pair<int64_t, string>(schemaId, field));
}

// Get attribute list
std::vector<std::pair<uint64_t, std::string>> Index::GetFields() { return fields_; }
}  // namespace mindrecord
}  // namespace mindspore
