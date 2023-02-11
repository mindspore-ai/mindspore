/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <regex>

namespace mindspore {
namespace lite {
bool EraseBlankSpaceAndLineBreak(std::string *input_string) {
  if (input_string == nullptr) {
    MS_LOG(ERROR) << "input_string is nullptr";
    return false;
  }
  (void)input_string->erase(0, input_string->find_first_not_of(" "));
  (void)input_string->erase(input_string->find_last_not_of(" ") + 1);
  (void)input_string->erase(std::remove(input_string->begin(), input_string->end(), '\r'), input_string->end());
  (void)input_string->erase(std::remove(input_string->begin(), input_string->end(), '\n'), input_string->end());
  return true;
}

bool EraseQuotes(std::string *input_string) {
  if (input_string == nullptr) {
    MS_LOG(ERROR) << "input_string is nullptr";
    return false;
  }
  if (!input_string->empty()) {
    std::string::size_type pos = 0;
    pos = input_string->find('\"', pos);
    while (pos != std::string::npos) {
      (void)input_string->erase(pos, 1);
      pos = input_string->find('\"', pos);
    }
  }
  return true;
}

bool FindAndReplaceAll(std::string *input_str, const std::string &search, const std::string &replace) {
  if (input_str == nullptr) {
    MS_LOG(ERROR) << "input_str is nullptr";
    return false;
  }
  auto pos = input_str->find(search);
  while (pos != std::string::npos) {
    input_str->replace(pos, search.size(), replace);
    pos = input_str->find(search, pos + replace.size());
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

std::vector<std::string> SplitStringToVector(const std::string &raw_str, const std::string &delimiter) {
  size_t pos_start = 0;
  size_t pos_end = 0;
  size_t delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = raw_str.find(delimiter, pos_start)) != std::string::npos) {
    token = raw_str.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }
  res.push_back(raw_str.substr(pos_start));
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
  std::regex r("^[[].+]$");
  if (!std::regex_match(str, r)) {
    MS_LOG(ERROR) << "the value should begin with [ and end with ]";
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

size_t Hex2ByteArray(const std::string &hex_str, unsigned char *byte_array, size_t max_len) {
  std::regex r("[0-9a-fA-F]+");
  if (!std::regex_match(hex_str, r)) {
    MS_LOG(ERROR) << "Some characters of dec_key not in [0-9a-fA-F]";
    return 0;
  }
  if (hex_str.size() % 2 == 1) {  // Mod 2 determines whether it is odd
    MS_LOG(ERROR) << "the hexadecimal dec_key length must be even";
    return 0;
  }
  size_t byte_len = hex_str.size() / 2;  // Two hexadecimal characters represent a byte
  if (byte_len > max_len) {
    MS_LOG(ERROR) << "the hexadecimal dec_key length exceeds the maximum limit: " << max_len;
    return 0;
  }
  constexpr int32_t a_val = 10;  // The value of 'A' in hexadecimal is 10
  constexpr size_t half_byte_offset = 4;
  for (size_t i = 0; i < byte_len; ++i) {
    size_t p = i * 2;  // The i-th byte is represented by the 2*i and 2*i+1 hexadecimal characters
    if (hex_str[p] >= 'a' && hex_str[p] <= 'f') {
      byte_array[i] = hex_str[p] - 'a' + a_val;
    } else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {
      byte_array[i] = hex_str[p] - 'A' + a_val;
    } else {
      byte_array[i] = hex_str[p] - '0';
    }
    if (hex_str[p + 1] >= 'a' && hex_str[p + 1] <= 'f') {
      byte_array[i] = (byte_array[i] << half_byte_offset) | (hex_str[p + 1] - 'a' + a_val);
    } else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {
      byte_array[i] = (byte_array[i] << half_byte_offset) | (hex_str[p + 1] - 'A' + a_val);
    } else {
      byte_array[i] = (byte_array[i] << half_byte_offset) | (hex_str[p + 1] - '0');
    }
  }
  return byte_len;
}
}  // namespace lite
}  // namespace mindspore
