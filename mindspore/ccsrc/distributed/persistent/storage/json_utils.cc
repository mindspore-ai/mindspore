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

#include "distributed/persistent/storage/json_utils.h"
#include "distributed/persistent/storage/file_io_utils.h"

namespace mindspore {
namespace distributed {
namespace storage {
bool JsonUtils::Initialize() {
  if (!FileIOUtils::IsFileOrDirExist(file_name_)) {
    FileIOUtils::CreateFile(file_name_);
    return true;
  }

  std::ifstream json_file(file_name_);
  try {
    json_file >> js_;
    json_file.close();
  } catch (nlohmann::json::exception &e) {
    json_file.close();
    std::string illegal_exception = e.what();
    MS_LOG(ERROR) << "Parse json file:" << file_name_ << " failed, the exception:" << illegal_exception;
    return false;
  }
  return true;
}

bool JsonUtils::Exists(const std::string &key) const {
  if (!js_.contains(key)) {
    return false;
  }
  return true;
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
