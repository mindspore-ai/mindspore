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

#include "mindrecord/include/shard_segment.h"
#include "common/utils.h"

#include "./securec.h"
#include "mindrecord/include/common/shard_utils.h"
#include "pybind11/pybind11.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::ERROR;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
ShardSegment::ShardSegment() { SetAllInIndex(false); }

std::pair<MSRStatus, vector<std::string>> ShardSegment::GetCategoryFields() {
  // Skip if already populated
  if (!candidate_category_fields_.empty()) return {SUCCESS, candidate_category_fields_};

  std::string sql = "PRAGMA table_info(INDEXES);";
  std::vector<std::vector<std::string>> field_names;

  char *errmsg = nullptr;
  int rc = sqlite3_exec(database_paths_[0], common::SafeCStr(sql), SelectCallback, &field_names, &errmsg);
  if (rc != SQLITE_OK) {
    MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
    sqlite3_free(errmsg);
    sqlite3_close(database_paths_[0]);
    return {FAILED, vector<std::string>{}};
  } else {
    MS_LOG(INFO) << "Get " << static_cast<int>(field_names.size()) << " records from index.";
  }

  uint32_t idx = kStartFieldId;
  while (idx < field_names.size()) {
    if (field_names[idx].size() < 2) {
      sqlite3_free(errmsg);
      sqlite3_close(database_paths_[0]);
      return {FAILED, vector<std::string>{}};
    }
    candidate_category_fields_.push_back(field_names[idx][1]);
    idx += 2;
  }
  sqlite3_free(errmsg);
  return {SUCCESS, candidate_category_fields_};
}

MSRStatus ShardSegment::SetCategoryField(std::string category_field) {
  if (GetCategoryFields().first != SUCCESS) {
    MS_LOG(ERROR) << "Get candidate category field failed";
    return FAILED;
  }
  category_field = category_field + "_0";
  if (std::any_of(std::begin(candidate_category_fields_), std::end(candidate_category_fields_),
                  [category_field](std::string x) { return x == category_field; })) {
    current_category_field_ = category_field;
    return SUCCESS;
  }
  MS_LOG(ERROR) << "Field " << category_field << " is not a candidate category field.";
  return FAILED;
}

std::pair<MSRStatus, std::string> ShardSegment::ReadCategoryInfo() {
  MS_LOG(INFO) << "Read category begin";
  auto ret = WrapCategoryInfo();
  if (ret.first != SUCCESS) {
    MS_LOG(ERROR) << "Get category info failed";
    return {FAILED, ""};
  }
  // Convert category info to json string
  auto category_json_string = ToJsonForCategory(ret.second);

  MS_LOG(INFO) << "Read category end";

  return {SUCCESS, category_json_string};
}

std::pair<MSRStatus, std::vector<std::tuple<int, std::string, int>>> ShardSegment::WrapCategoryInfo() {
  std::map<std::string, int> counter;

  std::string sql = "SELECT " + current_category_field_ + ", COUNT(" + current_category_field_ +
                    ") AS `value_occurrence` FROM indexes GROUP BY " + current_category_field_ + ";";

  for (auto &db : database_paths_) {
    std::vector<std::vector<std::string>> field_count;

    char *errmsg = nullptr;
    int rc = sqlite3_exec(db, common::SafeCStr(sql), SelectCallback, &field_count, &errmsg);
    if (rc != SQLITE_OK) {
      MS_LOG(ERROR) << "Error in select statement, sql: " << sql << ", error: " << errmsg;
      sqlite3_free(errmsg);
      sqlite3_close(db);
      return {FAILED, std::vector<std::tuple<int, std::string, int>>()};
    } else {
      MS_LOG(INFO) << "Get " << static_cast<int>(field_count.size()) << " records from index.";
    }

    for (const auto &field : field_count) {
      counter[field[0]] += std::stoi(field[1]);
    }
    sqlite3_free(errmsg);
  }

  int idx = 0;
  std::vector<std::tuple<int, std::string, int>> category_vec(counter.size());
  (void)std::transform(counter.begin(), counter.end(), category_vec.begin(), [&idx](std::tuple<std::string, int> item) {
    return std::make_tuple(idx++, std::get<0>(item), std::get<1>(item));
  });
  return {SUCCESS, std::move(category_vec)};
}

std::string ShardSegment::ToJsonForCategory(const std::vector<std::tuple<int, std::string, int>> &tri_vec) {
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

std::pair<MSRStatus, std::vector<std::vector<uint8_t>>> ShardSegment::ReadAtPageById(int64_t category_id,
                                                                                     int64_t page_no,
                                                                                     int64_t n_rows_of_page) {
  auto ret = WrapCategoryInfo();
  if (ret.first != SUCCESS) {
    MS_LOG(ERROR) << "Get category info";
    return {FAILED, std::vector<std::vector<uint8_t>>{}};
  }
  if (category_id >= static_cast<int>(ret.second.size()) || category_id < 0) {
    MS_LOG(ERROR) << "Illegal category id, id: " << category_id;
    return {FAILED, std::vector<std::vector<uint8_t>>{}};
  }
  int total_rows_in_category = std::get<2>(ret.second[category_id]);
  // Quit if category not found or page number is out of range
  if (total_rows_in_category <= 0 || page_no < 0 || n_rows_of_page <= 0 ||
      page_no * n_rows_of_page >= total_rows_in_category) {
    MS_LOG(ERROR) << "Illegal page no / page size, page no: " << page_no << ", page size: " << n_rows_of_page;
    return {FAILED, std::vector<std::vector<uint8_t>>{}};
  }

  std::vector<std::vector<uint8_t>> page;
  auto row_group_summary = ReadRowGroupSummary();

  uint64_t i_start = page_no * n_rows_of_page;
  uint64_t i_end = std::min(static_cast<int64_t>(total_rows_in_category), (page_no + 1) * n_rows_of_page);
  uint64_t idx = 0;
  for (const auto &rg : row_group_summary) {
    if (idx >= i_end) break;

    auto shard_id = std::get<0>(rg);
    auto group_id = std::get<1>(rg);
    auto details = ReadRowGroupCriteria(
      group_id, shard_id, std::make_pair(CleanUp(current_category_field_), std::get<1>(ret.second[category_id])));
    if (SUCCESS != std::get<0>(details)) {
      return {FAILED, std::vector<std::vector<uint8_t>>{}};
    }
    auto offsets = std::get<4>(details);
    uint64_t number_of_rows = offsets.size();
    if (idx + number_of_rows < i_start) {
      idx += number_of_rows;
      continue;
    }

    for (uint64_t i = 0; i < number_of_rows; ++i, ++idx) {
      if (idx >= i_start && idx < i_end) {
        auto ret1 = PackImages(group_id, shard_id, offsets[i]);
        if (SUCCESS != ret1.first) {
          return {FAILED, std::vector<std::vector<uint8_t>>{}};
        }
        page.push_back(std::move(ret1.second));
      }
    }
  }

  return {SUCCESS, std::move(page)};
}

std::pair<MSRStatus, std::vector<uint8_t>> ShardSegment::PackImages(int group_id, int shard_id,
                                                                    std::vector<uint64_t> offset) {
  const auto &ret = shard_header_->GetPageByGroupId(group_id, shard_id);
  if (SUCCESS != ret.first) {
    return {FAILED, std::vector<uint8_t>()};
  }
  const std::shared_ptr<Page> &blob_page = ret.second;

  // Pack image list
  std::vector<uint8_t> images(offset[1] - offset[0]);
  auto file_offset = header_size_ + page_size_ * (blob_page->GetPageID()) + offset[0];
  auto &io_seekg = file_streams_random_[0][shard_id]->seekg(file_offset, std::ios::beg);
  if (!io_seekg.good() || io_seekg.fail() || io_seekg.bad()) {
    MS_LOG(ERROR) << "File seekg failed";
    file_streams_random_[0][shard_id]->close();
    return {FAILED, {}};
  }

  auto &io_read = file_streams_random_[0][shard_id]->read(reinterpret_cast<char *>(&images[0]), offset[1] - offset[0]);
  if (!io_read.good() || io_read.fail() || io_read.bad()) {
    MS_LOG(ERROR) << "File read failed";
    file_streams_random_[0][shard_id]->close();
    return {FAILED, {}};
  }

  return {SUCCESS, std::move(images)};
}

std::pair<MSRStatus, std::vector<std::vector<uint8_t>>> ShardSegment::ReadAtPageByName(std::string category_name,
                                                                                       int64_t page_no,
                                                                                       int64_t n_rows_of_page) {
  auto ret = WrapCategoryInfo();
  if (ret.first != SUCCESS) {
    MS_LOG(ERROR) << "Get category info";
    return {FAILED, std::vector<std::vector<uint8_t>>{}};
  }
  for (const auto &categories : ret.second) {
    if (std::get<1>(categories) == category_name) {
      auto result = ReadAtPageById(std::get<0>(categories), page_no, n_rows_of_page);
      return result;
    }
  }

  return {FAILED, std::vector<std::vector<uint8_t>>()};
}

std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, json>>> ShardSegment::ReadAllAtPageById(
  int64_t category_id, int64_t page_no, int64_t n_rows_of_page) {
  auto ret = WrapCategoryInfo();
  if (ret.first != SUCCESS || category_id >= static_cast<int>(ret.second.size())) {
    MS_LOG(ERROR) << "Illegal category id, id: " << category_id;
    return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
  }
  int total_rows_in_category = std::get<2>(ret.second[category_id]);
  // Quit if category not found or page number is out of range
  if (total_rows_in_category <= 0 || page_no < 0 || page_no * n_rows_of_page >= total_rows_in_category) {
    MS_LOG(ERROR) << "Illegal page no: " << page_no << ", page size: " << n_rows_of_page;
    return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
  }

  std::vector<std::tuple<std::vector<uint8_t>, json>> page;
  auto row_group_summary = ReadRowGroupSummary();

  int i_start = page_no * n_rows_of_page;
  int i_end = std::min(static_cast<int64_t>(total_rows_in_category), (page_no + 1) * n_rows_of_page);
  int idx = 0;
  for (const auto &rg : row_group_summary) {
    if (idx >= i_end) break;

    auto shard_id = std::get<0>(rg);
    auto group_id = std::get<1>(rg);
    auto details = ReadRowGroupCriteria(
      group_id, shard_id, std::make_pair(CleanUp(current_category_field_), std::get<1>(ret.second[category_id])));
    if (SUCCESS != std::get<0>(details)) {
      return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
    }
    auto offsets = std::get<4>(details);
    auto labels = std::get<5>(details);

    int number_of_rows = offsets.size();
    if (idx + number_of_rows < i_start) {
      idx += number_of_rows;
      continue;
    }

    if (number_of_rows > static_cast<int>(labels.size())) {
      MS_LOG(ERROR) << "Illegal row number of page: " << number_of_rows;
      return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
    }
    for (int i = 0; i < number_of_rows; ++i, ++idx) {
      if (idx >= i_start && idx < i_end) {
        auto ret1 = PackImages(group_id, shard_id, offsets[i]);
        if (SUCCESS != ret1.first) {
          return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
        }
        page.emplace_back(std::move(ret1.second), std::move(labels[i]));
      }
    }
  }
  return {SUCCESS, std::move(page)};
}

std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, json>>> ShardSegment::ReadAllAtPageByName(
  std::string category_name, int64_t page_no, int64_t n_rows_of_page) {
  auto ret = WrapCategoryInfo();
  if (ret.first != SUCCESS) {
    MS_LOG(ERROR) << "Get category info";
    return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
  }

  // category_name to category_id
  int64_t category_id = -1;
  for (const auto &categories : ret.second) {
    std::string categories_name = std::get<1>(categories);

    if (categories_name == category_name) {
      category_id = std::get<0>(categories);
      break;
    }
  }

  if (category_id == -1) {
    return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, json>>{}};
  }

  return ReadAllAtPageById(category_id, page_no, n_rows_of_page);
}

std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>> ShardSegment::ReadAtPageByIdPy(
  int64_t category_id, int64_t page_no, int64_t n_rows_of_page) {
  auto res = ReadAllAtPageById(category_id, page_no, n_rows_of_page);
  if (res.first != SUCCESS) {
    return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>{}};
  }

  vector<std::tuple<std::vector<uint8_t>, pybind11::object>> json_data;
  std::transform(res.second.begin(), res.second.end(), std::back_inserter(json_data),
                 [](const std::tuple<std::vector<uint8_t>, json> &item) {
                   auto &j = std::get<1>(item);
                   pybind11::object obj = nlohmann::detail::FromJsonImpl(j);
                   return std::make_tuple(std::get<0>(item), std::move(obj));
                 });
  return {SUCCESS, std::move(json_data)};
}

std::pair<MSRStatus, std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>> ShardSegment::ReadAtPageByNamePy(
  std::string category_name, int64_t page_no, int64_t n_rows_of_page) {
  auto res = ReadAllAtPageByName(category_name, page_no, n_rows_of_page);
  if (res.first != SUCCESS) {
    return {FAILED, std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>{}};
  }
  vector<std::tuple<std::vector<uint8_t>, pybind11::object>> json_data;
  std::transform(res.second.begin(), res.second.end(), std::back_inserter(json_data),
                 [](const std::tuple<std::vector<uint8_t>, json> &item) {
                   auto &j = std::get<1>(item);
                   pybind11::object obj = nlohmann::detail::FromJsonImpl(j);
                   return std::make_tuple(std::get<0>(item), std::move(obj));
                 });
  return {SUCCESS, std::move(json_data)};
}

std::pair<ShardType, std::vector<std::string>> ShardSegment::GetBlobFields() {
  std::vector<std::string> blob_fields;
  for (auto &p : GetShardHeader()->GetSchemas()) {
    // assume one schema
    const auto &fields = p->GetBlobFields();
    blob_fields.assign(fields.begin(), fields.end());
    break;
  }
  return std::make_pair(kCV, blob_fields);
}

std::string ShardSegment::CleanUp(std::string field_name) {
  while (field_name.back() >= '0' && field_name.back() <= '9') field_name.pop_back();
  field_name.pop_back();
  return field_name;
}
}  // namespace mindrecord
}  // namespace mindspore
