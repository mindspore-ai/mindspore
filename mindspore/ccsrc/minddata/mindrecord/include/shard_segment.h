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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SEGMENT_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SEGMENT_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/shard_reader.h"

namespace mindspore {
namespace mindrecord {
using CATEGORY_INFO = std::vector<std::tuple<int, std::string, int>>;
using PAGES = std::vector<std::tuple<std::vector<uint8_t>, json>>;
using PAGES_LOAD = std::vector<std::tuple<std::vector<uint8_t>, pybind11::object>>;
using PAGES_WITH_BLOBS = std::vector<std::tuple<std::map<std::string, std::vector<uint8_t>>, json>>;
using PAGES_LOAD_WITH_BLOBS = std::vector<std::tuple<std::map<std::string, std::vector<uint8_t>>, pybind11::object>>;

class MINDRECORD_API ShardSegment : public ShardReader {
 public:
  ShardSegment();

  ~ShardSegment() override = default;

  /// \brief Get candidate category fields
  /// \return a list of fields names which are the candidates of category
  Status GetCategoryFields(std::shared_ptr<vector<std::string>> *fields_ptr);

  /// \brief Set category field
  /// \param[in] category_field category name
  /// \return true if category name is existed
  Status SetCategoryField(std::string category_field);

  /// \brief Thread-safe implementation of ReadCategoryInfo
  /// \return statistics data in json format with 2 field: "key" and "categories".
  ///         The value of "categories" is a list. Each Element in list is {count, id， name}
  ///              count: count of images in category
  ///              id:    internal unique identification, persistent
  ///              name:  category name
  ///  example:
  /// { "key": "label",
  ///    "categories": [ { "count": 3, "id": 0, "name": "sport", },
  ///                    { "count": 3, "id": 1, "name": "finance", } ] }
  Status ReadCategoryInfo(std::shared_ptr<std::string> *category_ptr);

  /// \brief Thread-safe implementation of ReadAtPageById
  /// \param[in] category_id category ID
  /// \param[in] page_no page number
  /// \param[in] n_rows_of_page rows number in one page
  /// \return images array, image is a vector of uint8_t
  Status ReadAtPageById(int64_t category_id, int64_t page_no, int64_t n_rows_of_page,
                        std::shared_ptr<std::vector<std::vector<uint8_t>>> *page_ptr);

  /// \brief Thread-safe implementation of ReadAtPageByName
  /// \param[in] category_name category Name
  /// \param[in] page_no page number
  /// \param[in] n_rows_of_page rows number in one page
  /// \return images array, image is a vector of uint8_t
  Status ReadAtPageByName(std::string category_name, int64_t page_no, int64_t n_rows_of_page,
                          std::shared_ptr<std::vector<std::vector<uint8_t>>> *pages_ptr);

  Status ReadAllAtPageById(int64_t category_id, int64_t page_no, int64_t n_rows_of_page,
                           std::shared_ptr<PAGES_WITH_BLOBS> *pages_ptr);

  Status ReadAllAtPageByName(std::string category_name, int64_t page_no, int64_t n_rows_of_page,
                             std::shared_ptr<PAGES_WITH_BLOBS> *pages_ptr);

  std::pair<ShardType, std::vector<std::string>> GetBlobFields();

 private:
  Status WrapCategoryInfo(std::shared_ptr<CATEGORY_INFO> *category_info_ptr);

  std::string ToJsonForCategory(const std::vector<std::tuple<int, std::string, int>> &tri_vec);

  std::string CleanUp(std::string fieldName);

  Status PackImages(int group_id, int shard_id, std::vector<uint64_t> offset,
                    std::shared_ptr<std::vector<uint8_t>> *images_ptr);

  std::vector<std::string> candidate_category_fields_;
  std::string current_category_field_;
  const uint32_t kStartFieldId = 9;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_SEGMENT_H_
