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

#include "ps/core/file_configuration.h"

namespace mindspore {
namespace ps {
namespace core {
bool FileConfiguration::Initialize() {
  if (!CommUtil::IsFileExists(file_path_)) {
    MS_LOG(INFO) << "The file path:" << file_path_ << " is not exist.";
    return false;
  }

  std::ifstream json_file(file_path_);
  try {
    json_file >> js;
    json_file.close();
    is_initialized_ = true;
  } catch (nlohmann::json::exception &e) {
    json_file.close();
    std::string illegal_exception = e.what();
    MS_LOG(ERROR) << "Parse json file:" << file_path_ << " failed, the exception:" << illegal_exception;
    return false;
  }
  return true;
}

bool FileConfiguration::IsInitialized() const { return is_initialized_.load(); }

std::string FileConfiguration::Get(const std::string &key, const std::string &defaultvalue) const {
  if (!js.contains(key)) {
    MS_LOG(WARNING) << "The key:" << key << " is not exist.";
    return defaultvalue;
  }
  std::string res = js.at(key).dump();
  return res;
}

std::string FileConfiguration::GetString(const std::string &key, const std::string &defaultvalue) const {
  if (!js.contains(key)) {
    MS_LOG(WARNING) << "The key:" << key << " is not exist.";
    return defaultvalue;
  }
  std::string res = js.at(key);
  return res;
}

int64_t FileConfiguration::GetInt(const std::string &key, int64_t default_value) const {
  if (!js.contains(key)) {
    MS_LOG(WARNING) << "The key:" << key << " is not exist.";
    return default_value;
  }
  int64_t res = js.at(key);
  return res;
}

void FileConfiguration::Put(const std::string &key, const std::string &value) {
  std::ofstream output_file(file_path_);
  js[key] = value;
  output_file << js.dump();

  output_file.close();
}

bool FileConfiguration::Exists(const std::string &key) const {
  if (!js.contains(key)) {
    return false;
  }
  return true;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
