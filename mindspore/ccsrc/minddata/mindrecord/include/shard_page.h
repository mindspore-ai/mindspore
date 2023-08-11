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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PAGE_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PAGE_H_

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/mindrecord/include/common/log_adapter.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/mindrecord_macro.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace mindrecord {
const std::string kPageTypeRaw = "RAW_DATA";
const std::string kPageTypeBlob = "BLOB_DATA";
const std::string kPageTypeNewColumn = "NEW_COLUMN_DATA";

class MINDRECORD_API Page {
 public:
  Page(const int &page_id, const int &shard_id, const std::string &page_type, const int &page_type_id,
       const uint64_t &start_row_id, const uint64_t end_row_id,
       const std::vector<std::pair<int, uint64_t>> &row_group_ids, const uint64_t page_size)
      : page_id_(page_id),
        shard_id_(shard_id),
        page_type_(page_type),
        page_type_id_(page_type_id),
        start_row_id_(start_row_id),
        end_row_id_(end_row_id),
        row_group_ids_(row_group_ids),
        page_size_(page_size) {}

  ~Page() = default;

  /// \brief get the page and its description
  /// \return the json format of the page and its description
  json GetPage() const;

  int GetPageID() const { return page_id_; }

  int GetShardID() const { return shard_id_; }

  int GetPageTypeID() const { return page_type_id_; }

  std::string GetPageType() const { return page_type_; }

  uint64_t GetPageSize() const { return page_size_; }

  uint64_t GetStartRowID() const { return start_row_id_; }

  uint64_t GetEndRowID() const { return end_row_id_; }

  void SetEndRowID(const uint64_t &end_row_id) { end_row_id_ = end_row_id; }

  void SetPageSize(const uint64_t &page_size) { page_size_ = page_size; }

  std::pair<int, uint64_t> GetLastRowGroupID() const { return row_group_ids_.back(); }

  std::vector<std::pair<int, uint64_t>> GetRowGroupIds() const { return row_group_ids_; }

  void SetRowGroupIds(const std::vector<std::pair<int, uint64_t>> &last_row_group_ids) {
    row_group_ids_ = last_row_group_ids;
  }

  void DeleteLastGroupId();

 private:
  int page_id_;
  int shard_id_;
  std::string page_type_;
  int page_type_id_;
  uint64_t start_row_id_;
  uint64_t end_row_id_;
  std::vector<std::pair<int, uint64_t>> row_group_ids_;
  uint64_t page_size_;
  // JSON page: {
  //            "page_id":X,
  //            "shard_id":X,
  //            "page_type":"XXX", (enum "raw_data", "blob_data", "new_column")
  //            "page_type_id":X,
  //            "start_row_id":X,
  //            "end_row_id":X,
  //            "row_group_ids":[{"id":X, "offset":X}],
  //            "page_size":X,
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_PAGE_H_
