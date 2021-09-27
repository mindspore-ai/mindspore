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

#include "minddata/mindrecord/include/shard_segment.h"
#include "utils/ms_utils.h"

#include "./securec.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "pybind11/pybind11.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
ShardSegment::ShardSegment() { SetAllInIndex(false); }

Status ShardSegment::GetCategoryFields(std::shared_ptr<vector<std::string>> *fields_ptr) {
  RETURN_UNEXPECTED_IF_NULL(fields_ptr);
  // Skip if already populated
  if (!candidate_category_fields_.empty()) {
    *fields_ptr = std::make_shared<vector<std::string>>(candidate_category_fields_);
    return Status::OK();
  }

  std::string sql = "PRAGMA table_info(INDEXES);";
  std::vector<std::vector<std::string>> field_names;

  char *errmsg = nullptr;
  int rc = sqlite3_exec(database_paths_[0], common::SafeCStr(sql), SelectCallback, &field_names, &errmsg);
  if (rc != SQLITE_OK) {
    std::ostringstream oss;
    oss << "Failed to execute sql [ " << common::SafeCStr(sql) << " ], " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(database_paths_[0]);
    database_paths_[0] = nullptr;
    RETURN_STATUS_UNEXPECTED(oss.str());
  } else {
    MS_LOG(INFO) << "Succeed to get " << static_cast<int>(field_names.size()) << " records from index.";
  }

  uint32_t idx = kStartFieldId;
  while (idx < field_names.size()) {
    if (field_names[idx].size() < 2) {
      sqlite3_free(errmsg);
      sqlite3_close(database_paths_[0]);
      database_paths_[0] = nullptr;
      RETURN_STATUS_UNEXPECTED("Invalid data, field_names size must be greater than 1, but got " +
                               std::to_string(field_names[idx].size()));
    }
    candidate_category_fields_.push_back(field_names[idx][1]);
    idx += 2;
  }
  sqlite3_free(errmsg);
  *fields_ptr = std::make_shared<vector<std::string>>(candidate_category_fields_);
  return Status::OK();
}

Status ShardSegment::SetCategoryField(std::string category_field) {
  std::shared_ptr<vector<std::string>> fields_ptr;
  RETURN_IF_NOT_OK(GetCategoryFields(&fields_ptr));
  category_field = category_field + "_0";
  if (std::any_of(std::begin(candidate_category_fields_), std::end(candidate_category_fields_),
                  [category_field](std::string x) { return x == category_field; })) {
    current_category_field_ = category_field;
    return Status::OK();
  }
  RETURN_STATUS_UNEXPECTED("Invalid data, field '" + category_field + "' is not a candidate category field.");
}

Status ShardSegment::ReadCategoryInfo(std::shared_ptr<std::string> *category_ptr) {
  RETURN_UNEXPECTED_IF_NULL(category_ptr);
  auto category_info_ptr = std::make_shared<CATEGORY_INFO>();
  RETURN_IF_NOT_OK(WrapCategoryInfo(&category_info_ptr));
  // Convert category info to json string
  *category_ptr = std::make_shared<std::string>(ToJsonForCategory(*category_info_ptr));

  return Status::OK();
}

Status ShardSegment::WrapCategoryInfo(std::shared_ptr<CATEGORY_INFO> *category_info_ptr) {
  RETURN_UNEXPECTED_IF_NULL(category_info_ptr);
  std::map<std::string, int> counter;
  CHECK_FAIL_RETURN_UNEXPECTED(ValidateFieldName(current_category_field_),
                               "Invalid data, field: " + current_category_field_ + "is invalid.");
  std::string sql = "SELECT " + current_category_field_ + ", COUNT(" + current_category_field_ +
                    ") AS `value_occurrence` FROM indexes GROUP BY " + current_category_field_ + ";";

  for (auto &db : database_paths_) {
    std::vector<std::vector<std::string>> field_count;

    char *errmsg = nullptr;
    if (sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &field_count, &errmsg) != SQLITE_OK) {
      std::ostringstream oss;
      oss << "Failed to execute sql [ " << common::SafeCStr(sql) << " ], " << errmsg;
      sqlite3_free(errmsg);
      sqlite3_close(db);
      db = nullptr;
      RETURN_STATUS_UNEXPECTED(oss.str());
    } else {
      MS_LOG(INFO) << "Succeed to get " << static_cast<int>(field_count.size()) << " records from index.";
    }

    for (const auto &field : field_count) {
      counter[field[0]] += std::stoi(field[1]);
    }
    sqlite3_free(errmsg);
  }

  int idx = 0;
  (*category_info_ptr)->resize(counter.size());
  (void)std::transform(
    counter.begin(), counter.end(), (*category_info_ptr)->begin(),
    [&idx](std::tuple<std::string, int> item) { return std::make_tuple(idx++, std::get<0>(item), std::get<1>(item)); });
  return Status::OK();
}

std::string ShardSegment::ToJsonForCategory(const CATEGORY_INFO &tri_vec) {
  std::vector<json> category_json_vec;
  for (auto q : tri_vec) {
    json j;
    j["id"] = std::get<0>(q);
    j["name"] = std::get<1>(q);
    j["count"] = std::get<2>(q);

    category_json_vec.emplace_back(j);
  }

  json j_vec(category_json_vec);
  json category_info;
  category_info["key"] = current_category_field_;
  category_info["categories"] = j_vec;
  return category_info.dump();
}

Status ShardSegment::ReadAtPageById(int64_t category_id, int64_t page_no, int64_t n_rows_of_page,
                                    std::shared_ptr<std::vector<std::vector<uint8_t>>> *page_ptr) {
  RETURN_UNEXPECTED_IF_NULL(page_ptr);
  auto category_info_ptr = std::make_shared<CATEGORY_INFO>();
  RETURN_IF_NOT_OK(WrapCategoryInfo(&category_info_ptr));
  CHECK_FAIL_RETURN_UNEXPECTED(category_id < static_cast<int>(category_info_ptr->size()) && category_id >= 0,
                               "Invalid data, category_id: " + std::to_string(category_id) +
                                 " must be in the range [0, " + std::to_string(category_info_ptr->size()) + "].");
  int total_rows_in_category = std::get<2>((*category_info_ptr)[category_id]);
  // Quit if category not found or page number is out of range
  CHECK_FAIL_RETURN_UNEXPECTED(total_rows_in_category > 0 && page_no >= 0 && n_rows_of_page > 0 &&
                                 page_no * n_rows_of_page < total_rows_in_category,
                               "Invalid data, page no: " + std::to_string(page_no) +
                                 "or page size: " + std::to_string(n_rows_of_page) + " is invalid.");

  auto row_group_summary = ReadRowGroupSummary();

  uint64_t i_start = page_no * n_rows_of_page;
  uint64_t i_end = std::min(static_cast<int64_t>(total_rows_in_category), (page_no + 1) * n_rows_of_page);
  uint64_t idx = 0;
  for (const auto &rg : row_group_summary) {
    if (idx >= i_end) break;

    auto shard_id = std::get<0>(rg);
    auto group_id = std::get<1>(rg);
    std::shared_ptr<ROW_GROUP_BRIEF> row_group_brief_ptr;
    RETURN_IF_NOT_OK(ReadRowGroupCriteria(
      group_id, shard_id,
      std::make_pair(CleanUp(current_category_field_), std::get<1>((*category_info_ptr)[category_id])), {""},
      &row_group_brief_ptr));
    auto offsets = std::get<3>(*row_group_brief_ptr);
    uint64_t number_of_rows = offsets.size();
    if (idx + number_of_rows < i_start) {
      idx += number_of_rows;
      continue;
    }

    for (uint64_t i = 0; i < number_of_rows; ++i, ++idx) {
      if (idx >= i_start && idx < i_end) {
        auto images_ptr = std::make_shared<std::vector<uint8_t>>();
        RETURN_IF_NOT_OK(PackImages(group_id, shard_id, offsets[i], &images_ptr));
        (*page_ptr)->push_back(std::move(*images_ptr));
      }
    }
  }

  return Status::OK();
}

Status ShardSegment::PackImages(int group_id, int shard_id, std::vector<uint64_t> offset,
                                std::shared_ptr<std::vector<uint8_t>> *images_ptr) {
  RETURN_UNEXPECTED_IF_NULL(images_ptr);
  std::shared_ptr<Page> page_ptr;
  RETURN_IF_NOT_OK(shard_header_->GetPageByGroupId(group_id, shard_id, &page_ptr));
  // Pack image list
  (*images_ptr)->resize(offset[1] - offset[0]);

  auto file_offset = header_size_ + page_size_ * page_ptr->GetPageID() + offset[0];
  auto &io_seekg = file_streams_random_[0][shard_id]->seekg(file_offset, std::ios::beg);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    file_streams_random_[0][shard_id]->close();
    RETURN_STATUS_UNEXPECTED("Failed to seekg file.");
  }

  auto &io_read =
    file_streams_random_[0][shard_id]->read(reinterpret_cast<char *>(&((*(*images_ptr))[0])), offset[1] - offset[0]);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    file_streams_random_[0][shard_id]->close();
    RETURN_STATUS_UNEXPECTED("Failed to read file.");
  }
  return Status::OK();
}

Status ShardSegment::ReadAtPageByName(std::string category_name, int64_t page_no, int64_t n_rows_of_page,
                                      std::shared_ptr<std::vector<std::vector<uint8_t>>> *pages_ptr) {
  RETURN_UNEXPECTED_IF_NULL(pages_ptr);
  auto category_info_ptr = std::make_shared<CATEGORY_INFO>();
  RETURN_IF_NOT_OK(WrapCategoryInfo(&category_info_ptr));
  for (const auto &categories : *category_info_ptr) {
    if (std::get<1>(categories) == category_name) {
      RETURN_IF_NOT_OK(ReadAtPageById(std::get<0>(categories), page_no, n_rows_of_page, pages_ptr));
      return Status::OK();
    }
  }

  RETURN_STATUS_UNEXPECTED("category_name: " + category_name + " could not found.");
}

Status ShardSegment::ReadAllAtPageById(int64_t category_id, int64_t page_no, int64_t n_rows_of_page,
                                       std::shared_ptr<PAGES> *pages_ptr) {
  RETURN_UNEXPECTED_IF_NULL(pages_ptr);
  auto category_info_ptr = std::make_shared<CATEGORY_INFO>();
  RETURN_IF_NOT_OK(WrapCategoryInfo(&category_info_ptr));
  CHECK_FAIL_RETURN_UNEXPECTED(category_id < static_cast<int64_t>(category_info_ptr->size()),
                               "Invalid data, category_id: " + std::to_string(category_id) +
                                 " must be in the range [0, " + std::to_string(category_info_ptr->size()) + "].");

  int total_rows_in_category = std::get<2>((*category_info_ptr)[category_id]);
  // Quit if category not found or page number is out of range
  CHECK_FAIL_RETURN_UNEXPECTED(total_rows_in_category > 0 && page_no >= 0 && n_rows_of_page > 0 &&
                                 page_no * n_rows_of_page < total_rows_in_category,
                               "Invalid data, page no: " + std::to_string(page_no) +
                                 "or page size: " + std::to_string(n_rows_of_page) + " is invalid.");
  auto row_group_summary = ReadRowGroupSummary();

  int i_start = page_no * n_rows_of_page;
  int i_end = std::min(static_cast<int64_t>(total_rows_in_category), (page_no + 1) * n_rows_of_page);
  int idx = 0;
  for (const auto &rg : row_group_summary) {
    if (idx >= i_end) {
      break;
    }

    auto shard_id = std::get<0>(rg);
    auto group_id = std::get<1>(rg);
    std::shared_ptr<ROW_GROUP_BRIEF> row_group_brief_ptr;
    RETURN_IF_NOT_OK(ReadRowGroupCriteria(
      group_id, shard_id,
      std::make_pair(CleanUp(current_category_field_), std::get<1>((*category_info_ptr)[category_id])), {""},
      &row_group_brief_ptr));
    auto offsets = std::get<3>(*row_group_brief_ptr);
    auto labels = std::get<4>(*row_group_brief_ptr);

    int number_of_rows = offsets.size();
    if (idx + number_of_rows < i_start) {
      idx += number_of_rows;
      continue;
    }
    CHECK_FAIL_RETURN_UNEXPECTED(number_of_rows <= static_cast<int>(labels.size()),
                                 "Invalid data, number_of_rows: " + std::to_string(number_of_rows) + " is invalid.");
    for (int i = 0; i < number_of_rows; ++i, ++idx) {
      if (idx >= i_start && idx < i_end) {
        auto images_ptr = std::make_shared<std::vector<uint8_t>>();
        RETURN_IF_NOT_OK(PackImages(group_id, shard_id, offsets[i], &images_ptr));
        (*pages_ptr)->emplace_back(std::move(*images_ptr), std::move(labels[i]));
      }
    }
  }
  return Status::OK();
}

Status ShardSegment::ReadAllAtPageByName(std::string category_name, int64_t page_no, int64_t n_rows_of_page,
                                         std::shared_ptr<PAGES> *pages_ptr) {
  RETURN_UNEXPECTED_IF_NULL(pages_ptr);
  auto category_info_ptr = std::make_shared<CATEGORY_INFO>();
  RETURN_IF_NOT_OK(WrapCategoryInfo(&category_info_ptr));
  // category_name to category_id
  int64_t category_id = -1;
  for (const auto &categories : *category_info_ptr) {
    std::string categories_name = std::get<1>(categories);

    if (categories_name == category_name) {
      category_id = std::get<0>(categories);
      break;
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(category_id != -1, "category_name: " + category_name + " could not found.");
  return ReadAllAtPageById(category_id, page_no, n_rows_of_page, pages_ptr);
}

std::pair<ShardType, std::vector<std::string>> ShardSegment::GetBlobFields() {
  std::vector<std::string> blob_fields;
  auto schema_list = GetShardHeader()->GetSchemas();
  if (!schema_list.empty()) {
    const auto &fields = schema_list[0]->GetBlobFields();
    blob_fields.assign(fields.begin(), fields.end());
  }
  return std::make_pair(kCV, blob_fields);
}

std::string ShardSegment::CleanUp(std::string field_name) {
  while (field_name.back() >= '0' && field_name.back() <= '9') {
    field_name.pop_back();
  }
  field_name.pop_back();
  return field_name;
}
}  // namespace mindrecord
}  // namespace mindspore
