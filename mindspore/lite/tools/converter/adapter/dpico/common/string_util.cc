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

#include "common/string_util.h"
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include "mindapi/base/logging.h"

namespace mindspore {
namespace dpico {
int EraseBlankSpace(std::string *input_string) {
  if (input_string == nullptr) {
    MS_LOG(ERROR) << "input string is nullptr";
    return RET_ERROR;
  }
  if (!input_string->empty()) {
    std::string::size_type pos = 0;
    pos = input_string->find(' ', pos);
    while (pos != std::string::npos) {
      (void)input_string->erase(pos, 1);
      pos = input_string->find(' ', pos);
    }
  }
  return RET_OK;
}

int EraseHeadTailSpace(std::string *input_string) {
  if (input_string == nullptr) {
    MS_LOG(ERROR) << "input string is nullptr";
    return RET_ERROR;
  }
  if (!input_string->empty()) {
    (void)input_string->erase(0, input_string->find_first_not_of(' '));
    (void)input_string->erase(input_string->find_last_not_of(' ') + 1);
  }
  return RET_OK;
}

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

std::string RemoveSpecifiedChar(const std::string &origin_str, char specified_ch) {
  std::string res = origin_str;
  (void)res.erase(std::remove(res.begin(), res.end(), specified_ch), res.end());
  return res;
}

std::string ReplaceSpecifiedChar(const std::string &origin_str, char origin_ch, char target_ch) {
  std::string res = origin_str;
  std::replace(res.begin(), res.end(), origin_ch, target_ch);
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
}  // namespace dpico
}  // namespace mindspore
