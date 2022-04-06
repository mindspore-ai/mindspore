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

#include "common/string_util.h"
#include <vector>
#include <string>
#include <algorithm>
#include "common/log_util.h"

namespace mindspore {
namespace lite {
std::vector<std::string> SplitString(const std::string &raw_str, char delimiter) {
  if (raw_str.empty()) {
    MS_LOG(ERROR) << "input string is empty.";
    return {};
  }
  std::vector<std::string> res;
  std::string::size_type last_pos = 0;
  auto cur_pos = raw_str.find(delimiter);
  while (cur_pos != std::string::npos) {
    res.push_back(raw_str.substr(last_pos, cur_pos - last_pos));
    cur_pos++;
    last_pos = cur_pos;
    cur_pos = raw_str.find(delimiter, cur_pos);
  }
  if (last_pos < raw_str.size()) {
    res.push_back(raw_str.substr(last_pos, raw_str.size() - last_pos + 1));
  }
  return res;
}

bool IsValidUnsignedNum(const std::string &num_str) {
  return !num_str.empty() && std::all_of(num_str.begin(), num_str.end(), ::isdigit);
}

bool IsValidDoubleNum(const std::string &num_str) {
  if (num_str.empty()) {
    return false;
  }
  std::istringstream iss(num_str);
  double d;
  iss >> std::noskipws >> d;
  return iss.eof() && !iss.fail();
}
}  // namespace lite
}  // namespace mindspore
