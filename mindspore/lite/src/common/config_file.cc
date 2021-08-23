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

namespace mindspore {
namespace lite {
std::map<std::string, std::string> GetSectionInfoFromConfigFile(const std::string &file,
                                                                const std::string &section_name) {
  std::map<std::string, std::string> section_info;
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return section_info;
  }
  auto resolved_path = std::make_unique<char[]>(PATH_MAX);
  if (resolved_path == nullptr) {
    MS_LOG(ERROR) << "new resolved_path failed";
    return section_info;
  }

#ifdef _WIN32
  char *real_path = _fullpath(resolved_path.get(), file.c_str(), MAX_CONFIG_FILE_LENGTH);
#else
  char *real_path = realpath(file.c_str(), resolved_path.get());
#endif
  if (real_path == nullptr || strlen(real_path) == 0) {
    MS_LOG(ERROR) << "file path is not valid : " << file;
    return section_info;
  }
  std::ifstream ifs(resolved_path.get());
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return section_info;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << "open failed";
    return section_info;
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
      std::string section = line.substr(1, line.length() - 2);
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
      section_info.insert(std::make_pair(key, value));
    }
  }

  ifs.close();
  return section_info;
}
}  // namespace lite
}  // namespace mindspore
