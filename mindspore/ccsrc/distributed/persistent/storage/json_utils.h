/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_JSON_UTILS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_JSON_UTILS_H_

#include <fstream>
#include <string>

#include "include/common/utils/json_operation_utils.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace distributed {
namespace storage {
// This class uses json format to store and obtain a large number of key-value pairs, supports creating or opening json
// files, reading or modifying key-value in json.
class JsonUtils {
 public:
  explicit JsonUtils(const std::string &file_name) : file_name_(file_name) {}
  ~JsonUtils() = default;

  // Load or create a json file.
  bool Initialize();

  // Get the value corresponding to the key in json.
  template <typename T>
  T Get(const std::string &key) const;

  // Insert a key-value pair into json or change the value corresponding to the key in json.
  template <typename T>
  void Insert(const std::string &key, const T &value);

  // Check whether key exists in json or not.
  bool Exists(const std::string &key) const;

 private:
  // Json object.
  nlohmann::json js_;

  // The json file path.
  std::string file_name_;
};

template <typename T>
T JsonUtils::Get(const std::string &key) const {
  if (!js_.contains(key)) {
    MS_LOG(EXCEPTION) << "The key:" << key << " is not exist.";
  }

  return GetJsonValue<T>(js_, key);
}

template <typename T>
void JsonUtils::Insert(const std::string &key, const T &value) {
  std::ofstream output_file(file_name_);
  js_[key] = value;
  output_file << js_.dump();
  output_file.close();
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_JSON_UTILS_H_
