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

#include "tools/common/parse_config_utils.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/tools/common/string_util.h"
#include "src/common/log_adapter.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
int ReadFileToIfstream(const std::string &file_path, std::ifstream *ifstream) {
  if (ifstream == nullptr) {
    MS_LOG(ERROR) << "ifstream is nullptr.";
    return RET_ERROR;
  }
  if (file_path.empty()) {
    MS_LOG(ERROR) << "file path is empty.";
    return RET_ERROR;
  }
  std::string real_path = RealPath(file_path.c_str());
  if (real_path.empty()) {
    MS_LOG(ERROR) << "get real path failed.";
    return RET_ERROR;
  }
  ifstream->open(real_path, std::ios::in);
  if (!ifstream->good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist.";
    return RET_ERROR;
  }
  if (!ifstream->is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitLineToMap(std::ifstream *ifs, std::map<std::string, std::map<std::string, std::string>> *maps,
                   std::map<int, std::map<std::string, std::string>> *model_param_infos, const char &ignore_delimiter,
                   const char &split_delimiter) {
  if (ifs == nullptr || maps == nullptr) {
    MS_LOG(ERROR) << "ifs or maps is nullptr.";
    return RET_ERROR;
  }
  std::string raw_line;
  size_t num_of_line = 0;
  const size_t kMaxLineCount = 9999;
  std::string section = "DEFAULT";
  int model_index = -1;
  while (std::getline(*ifs, raw_line)) {
    if (num_of_line > kMaxLineCount) {
      MS_LOG(ERROR) << "the line count is exceeds the maximum range 9999.";
      return RET_ERROR;
    }
    if (raw_line.empty() || raw_line.at(0) == ignore_delimiter) {
      continue;
    }
    num_of_line++;

    if (!EraseBlankSpaceAndLineBreak(&raw_line)) {
      MS_LOG(ERROR) << "Erase Blank Space failed.";
      return RET_ERROR;
    }

    // remove value quotes eg: "/mnt/image" -> /mnt/image
    if (!EraseQuotes(&raw_line)) {
      MS_LOG(ERROR) << "Erase Quotes failed.";
      return RET_ERROR;
    }

    if (raw_line.empty()) {
      continue;
    }
    if (raw_line.at(0) == '[') {
      section = raw_line.substr(1, raw_line.size() - 2);
      if (section == "model_param") {
        ++model_index;
      }
      continue;
    }
    auto split_vector = SplitStringToVector(raw_line, split_delimiter);
    if (split_vector.size() != 2) {
      MS_LOG(ERROR) << "split vector size != 2";
      return RET_ERROR;
    }
    std::string key = split_vector.at(0);
    if (!EraseBlankSpaceAndLineBreak(&key)) {
      MS_LOG(ERROR) << "Erase Blank Space for key failed.";
      return RET_ERROR;
    }
    std::string value = split_vector.at(1);
    if (!EraseBlankSpaceAndLineBreak(&value)) {
      MS_LOG(ERROR) << "Erase Blank Space for value failed.";
      return RET_ERROR;
    }
    if (section == "model_param") {
      if (model_param_infos != nullptr) {
        (*model_param_infos)[model_index][key] = value;
      }
    } else {
      (*maps)[section][key] = value;
    }
  }
  return RET_OK;
}

int ParseConfigFile(const std::string &config_file_path,
                    std::map<std::string, std::map<std::string, std::string>> *maps,
                    std::map<int, std::map<std::string, std::string>> *model_param_infos) {
  auto real_path = RealPath(config_file_path.c_str());
  if (real_path.empty()) {
    MS_LOG(ERROR) << "real path is invalid.";
    return RET_ERROR;
  }
  std::ifstream ifs;
  auto ret = ReadFileToIfstream(config_file_path, &ifs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed.";
    return ret;
  }
  ret = SplitLineToMap(&ifs, maps, model_param_infos, '#', '=');
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split line to map failed.";
    ifs.close();
    return ret;
  }
  ifs.close();
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
