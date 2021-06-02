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
    MS_LOG(ERROR) << "The file path:" << file_path_ << " is not exist.";
    return false;
  }

  try {
    std::ifstream json_file(file_path_);
    json_file >> js;
    json_file.close();
  } catch (nlohmann::json::exception &e) {
    std::string illegal_exception = e.what();
    MS_LOG(ERROR) << "Parse json file:" << file_path_ << " failed, the exception:" << illegal_exception;
    return false;
  }
  return true;
}

std::string FileConfiguration::Get(const std::string &key, const std::string &defaultvalue) const {
  if (!js.contains(key)) {
    MS_LOG(WARNING) << "The key:" << key << " is not exit.";
    return defaultvalue;
  }
  std::string res = js.at(key);
  return res;
}

void FileConfiguration::Put(const std::string &key, const std::string &value) {
  std::ofstream output_file(file_path_);
  js[key] = value;
  output_file << js.dump();

  output_file.close();
}

}  // namespace core
}  // namespace ps
}  // namespace mindspore
