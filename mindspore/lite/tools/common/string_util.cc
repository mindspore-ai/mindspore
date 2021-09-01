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

#include "tools/common/string_util.h"
#include <algorithm>
#include <vector>
#include <string>

namespace mindspore {
namespace lite {
int EraseBlankSpaceAndLineBreak(std::string *input_string) {
  if (input_string == nullptr) {
    MS_LOG(ERROR) << "input_string is nullptr";
    return false;
  }
  input_string->erase(std::remove(input_string->begin(), input_string->end(), ' '), input_string->end());
  input_string->erase(std::remove(input_string->begin(), input_string->end(), '\r'), input_string->end());
  input_string->erase(std::remove(input_string->begin(), input_string->end(), '\n'), input_string->end());
  return true;
}

int EraseQuotes(std::string *input_string) {
  if (input_string == nullptr) {
    MS_LOG(ERROR) << "input_string is nullptr";
    return false;
  }
  if (!input_string->empty()) {
    std::string::size_type pos = 0;
    pos = input_string->find('\"', pos);
    while (pos != std::string::npos) {
      input_string->erase(pos, 1);
      pos = input_string->find('\"', pos);
    }
  }
  return true;
}

std::vector<std::string> SplitStringToVector(const std::string &raw_str, const char &delimiter) {
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

bool ConvertIntNum(const std::string &str, int *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return false;
  }
  char *ptr = nullptr;
  constexpr int kBase = 10;
  *value = strtol(str.c_str(), &ptr, kBase);
  return ptr == (str.c_str() + str.size());
}

bool ConvertDoubleNum(const std::string &str, double *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return false;
  }
  char *endptr = nullptr;
  *value = strtod(str.c_str(), &endptr);
  return *str.c_str() != 0 && *endptr == 0;
}

bool ConvertBool(std::string str, bool *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return false;
  }
  std::transform(str.begin(), str.end(), str.begin(), ::tolower);
  if (str == "true") {
    *value = true;
    return true;
  } else if (str == "false") {
    *value = false;
    return true;
  } else {
    return false;
  }
}

bool ConvertDoubleVector(const std::string &str, std::vector<double> *value) {
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return false;
  }
  // remove []
  std::string tmp_str = str.substr(1, str.size() - 2);
  // split ,
  std::vector<std::string> strings = SplitStringToVector(tmp_str, ',');
  size_t size = strings.size();
  value->resize(size);
  // ConvertToFloatVec
  for (size_t i = 0; i < size; i++) {
    if (!ConvertDoubleNum(strings[i], &value->at(i))) {
      MS_LOG(ERROR) << "Invalid num";
      return false;
    }
  }
  return true;
}
}  // namespace lite
}  // namespace mindspore
