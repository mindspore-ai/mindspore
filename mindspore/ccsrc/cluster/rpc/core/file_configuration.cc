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

#include "include/cluster/rpc/file_configuration.h"

namespace mindspore {
namespace ps {
namespace core {
bool FileConfiguration::Initialize() {
  if (!CommUtil::IsFileExists(file_path_)) {
    MS_LOG(INFO) << "The file:" << file_path_ << " is not exist.";

    if (CommUtil::CreateDirectory(file_path_)) {
      MS_LOG(INFO) << "Create directory for file:" << file_path_ << " success.";
    }
    return false;
  }

  if (!CommUtil::IsFileReadable(file_path_)) {
    MS_LOG(EXCEPTION) << "The file path: " << file_path_ << " is not readable.";
  }

  if (CommUtil::IsFileEmpty(file_path_)) {
    MS_LOG(EXCEPTION) << "The file path: " << file_path_ << " content is empty.";
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

std::vector<nlohmann::json> FileConfiguration::GetVector(const std::string &key) const {
  if (!js.contains(key)) {
    MS_LOG(WARNING) << "The key:" << key << " is not exist.";
    return std::vector<nlohmann::json>();
  }

  return js.at(key);
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

void FileConfiguration::Put(const std::string &key, const std::string &value) { js[key] = value; }

bool FileConfiguration::Exists(const std::string &key) const {
  if (!js.contains(key)) {
    return false;
  }
  return true;
}

std::string FileConfiguration::file_path() const { return file_path_; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
