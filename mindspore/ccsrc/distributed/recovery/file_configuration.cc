/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <fstream>
#include "utils/log_adapter.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "distributed/recovery/file_configuration.h"

namespace mindspore {
namespace distributed {
namespace recovery {
bool FileConfiguration::Initialize() {
  // If there is no local file, create an empty one.
  if (!storage::FileIOUtils::IsFileOrDirExist(file_)) {
    storage::FileIOUtils::CreateFile(file_);
    RETURN_IF_FALSE_WITH_LOG(storage::FileIOUtils::IsFileOrDirExist(file_),
                             "Failed to create the local configuration file " + file_);
    // There is already an existing local file, load and parse the values.
  } else {
    std::ifstream in_stream(file_);
    try {
      in_stream >> values_;
      in_stream.close();
    } catch (nlohmann::json::exception &e) {
      in_stream.close();
      std::string exception = e.what();
      MS_LOG(ERROR) << "Failed to parse the existing local file: " << file_ << ", the exception: " << exception;
      return false;
    }
  }
  return true;
}

std::string FileConfiguration::Get(const std::string &key, const std::string &defaultvalue) const {
  if (!values_.contains(key)) {
    return defaultvalue;
  }
  return values_.at(key);
}

void FileConfiguration::Put(const std::string &key, const std::string &value) { values_[key] = value; }

bool FileConfiguration::Exists(const std::string &key) const { return values_.contains(key); }

bool FileConfiguration::Empty() const { return values_.size() == 0; }

bool FileConfiguration::Flush() {
  if (!storage::FileIOUtils::IsFileOrDirExist(file_)) {
    MS_LOG(EXCEPTION) << "The local configuration file : " << file_ << " does not exist.";
  }
  // Write all the configuration items into local file.
  std::ofstream output_file(file_);
  output_file << values_.dump();
  output_file.close();

  return true;
}
}  // namespace recovery
}  // namespace distributed
}  // namespace mindspore
