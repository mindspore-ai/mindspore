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

#include "minddata/mindrecord/include/shard_page.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace mindrecord {
json Page::GetPage() const {
  json str_page;
  str_page["page_id"] = page_id_;
  str_page["shard_id"] = shard_id_;
  str_page["page_type"] = page_type_;
  str_page["page_type_id"] = page_type_id_;
  str_page["start_row_id"] = start_row_id_;
  str_page["end_row_id"] = end_row_id_;
  if (row_group_ids_.size() == 0) {
    json row_groups = json({});
    row_groups["id"] = 0;
    row_groups["offset"] = 0;
    str_page["row_group_ids"].push_back(row_groups);
  } else {
    for (const auto &rg : row_group_ids_) {
      json row_groups = json({});
      row_groups["id"] = rg.first;
      row_groups["offset"] = rg.second;
      str_page["row_group_ids"].push_back(row_groups);
    }
  }
  str_page["page_size"] = page_size_;
  return str_page;
}

void Page::DeleteLastGroupId() {
  if (!row_group_ids_.empty()) {
    page_size_ = row_group_ids_.back().second;
    row_group_ids_.pop_back();
  }
}
}  // namespace mindrecord
}  // namespace mindspore
