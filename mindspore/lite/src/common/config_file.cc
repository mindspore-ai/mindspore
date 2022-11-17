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
constexpr size_t kMinSectionLineLength = 2;
constexpr size_t kMaxValidLineCount = 100000;
constexpr size_t kMaxLineCount = 100100;
}  // namespace

namespace mindspore {
namespace lite {
namespace {
void ParseLine(const std::string &line, std::map<std::string, std::string> *section_config, std::string *section,
               size_t *valid_line_count, std::map<std::string, std::map<std::string, std::string>> *config) {
  // eg: [section]
  //     key=value
  if (line[0] == '[' && line[line.length() - 1] == ']') {
    if (!section->empty() && !section_config->empty()) {
      (void)config->insert(std::make_pair(*section, *section_config));
    }
    section_config->clear();
    *section = line.substr(1, line.length() - kLengthOfParentheses);
    *valid_line_count = *valid_line_count + 1;
  }

  if (!section->empty()) {
    auto index = line.find('=');
    if (index == std::string::npos) {
      return;
    }
    auto key = line.substr(0, index);
    if (index + 1 > line.size()) {
      return;
    }
    auto value = line.substr(index + 1);
    lite::Trim(&key);
    lite::Trim(&value);
    (void)section_config->insert(std::make_pair(key, value));
    *valid_line_count = *valid_line_count + 1;
  }
}
}  // namespace

int GetAllSectionInfoFromConfigFile(const std::string &file, ConfigInfos *config) {
  if (file.empty() || config == nullptr) {
    MS_LOG(ERROR) << "input Invalid!check file and config.";
    return RET_ERROR;
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path fail!";
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
  std::string section;
  std::map<std::string, std::string> section_config;
  size_t line_count = 0;
  size_t valid_line_count = 0;
  while (std::getline(ifs, line)) {
    line_count++;
    if (line_count >= kMaxLineCount || valid_line_count >= kMaxValidLineCount) {
      MS_LOG(ERROR) << "config too many lines!";
      ifs.close();
      return RET_ERROR;
    }
    lite::Trim(&line);
    if (line.length() <= kMinSectionLineLength || line[0] == '#') {
      continue;
    }
    ParseLine(line, &section_config, &section, &valid_line_count, config);
  }
  if (!section.empty() && !section_config.empty()) {
    (void)config->insert(std::make_pair(section, section_config));
  }
  ifs.close();
  return RET_OK;
}

void ParserExecutionPlan(const std::map<std::string, std::string> *config_infos,
                         std::map<std::string, TypeId> *data_type_plan) {
  for (auto info : *config_infos) {
    std::string op_name = info.first;
    std::string value = info.second;
    if (value.empty()) {
      MS_LOG(WARNING) << "Empty info in execution_plan";
      continue;
    }
    if (value[0] == '"' && value[value.length() - 1] == '"') {
      value = value.substr(1, value.length() - kLengthOfParentheses);
    }
    auto index = value.find(':');
    if (index == std::string::npos) {
      MS_LOG(WARNING) << "Invalid info in execution_plan: " << value;
      continue;
    }
    auto data_type_key = value.substr(0, index);
    if (index + 1 > value.size()) {
      return;
    }
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
    (void)data_type_plan->insert(std::make_pair(op_name, type_id));
  }
}
}  // namespace lite
}  // namespace mindspore
