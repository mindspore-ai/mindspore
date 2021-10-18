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

#include "src/common/config_file.h"

#ifdef _MSC_VER
#define PATH_MAX 1024
#endif
namespace {
constexpr size_t kLengthOfParentheses = 2;
}

namespace mindspore {
namespace lite {
int GetSectionInfoFromConfigFile(const std::string &file, const std::string &section_name,
                                 std::map<std::string, std::string> *section_info) {
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return RET_ERROR;
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return RET_ERROR;
  }

#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), file.c_str(), MAX_CONFIG_FILE_LENGTH);
#else
  char *real_path = realpath(file.c_str(), resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << file;
    return RET_ERROR;
  }
  std::ifstream ifs(resolved_path.get());
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return RET_ERROR;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << "open failed";
    return RET_ERROR;
  }
  std::string line;

  bool find_section = false;
  while (std::getline(ifs, line)) {
    lite::Trim(&line);
    if (line.empty()) {
      continue;
    }
    if (line[0] == '#') {
      continue;
    }

    if (line[0] == '[') {
      if (find_section == true) {
        break;
      }
      std::string section = line.substr(1, line.length() - kLengthOfParentheses);
      if (section != section_name) {
        continue;
      }
      find_section = true;
    }

    if (find_section == true) {
      auto index = line.find('=');
      if (index == std::string::npos) {
        continue;
      }
      auto key = line.substr(0, index);
      auto value = line.substr(index + 1);
      lite::Trim(&key);
      lite::Trim(&value);
      section_info->insert(std::make_pair(key, value));
    }
  }

  ifs.close();
  return RET_OK;
}

void ParserExecutionPlan(const std::map<std::string, std::string> *config_infos,
                         std::map<std::string, TypeId> *data_type_plan) {
  for (auto info : *config_infos) {
    std::string op_name = info.first;
    std::string value = info.second;
    if (value[0] == '"' && value[value.length() - 1] == '"') {
      value = value.substr(1, value.length() - kLengthOfParentheses);
    }
    auto index = value.find(':');
    if (index == std::string::npos) {
      MS_LOG(WARNING) << "Invalid info in execution_plan: " << value;
      continue;
    }
    auto data_type_key = value.substr(0, index);
    auto data_type_value = value.substr(index + 1);
    if (data_type_key != "data_type") {
      MS_LOG(WARNING) << "Invalid key in execution_plan: " << value;
      continue;
    }
    TypeId type_id = kTypeUnknown;
    if (data_type_value == "float32") {
      type_id = kNumberTypeFloat32;
    } else if (data_type_value == "float16") {
      type_id = kNumberTypeFloat16;
    } else {
      MS_LOG(WARNING) << "Invalid value in execution_plan: " << value;
      continue;
    }
    data_type_plan->insert(std::make_pair(op_name, type_id));
  }
}
}  // namespace lite
}  // namespace mindspore
