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
#include "debug/utils.h"
#include "mindspore/core/utils/log_adapter.h"

namespace mindspore {
bool CheckStoull(uint64_t *const output_digit, const std::string &input_str) {
  try {
    *output_digit = std::stoull(input_str);
  } catch (const std::out_of_range &oor) {
    MS_LOG(ERROR) << "Out of Range error: " << oor.what() << " when parse " << input_str;
    return false;
  } catch (const std::invalid_argument &ia) {
    MS_LOG(ERROR) << "Invalid argument: " << ia.what() << " when parse " << input_str;
    return false;
  }
  return true;
}

bool CheckStoul(size_t *const output_digit, const std::string &input_str) {
  try {
    *output_digit = std::stoul(input_str);
  } catch (const std::out_of_range &oor) {
    MS_LOG(ERROR) << "Out of Range error: " << oor.what() << " when parse " << input_str;
    return false;
  } catch (const std::invalid_argument &ia) {
    MS_LOG(ERROR) << "Invalid argument: " << ia.what() << " when parse " << input_str;
    return false;
  }
  return true;
}

bool CheckStoi(int64_t *const output_digit, const std::string &input_str) {
  try {
    *output_digit = std::stoi(input_str);
  } catch (const std::out_of_range &oor) {
    MS_LOG(ERROR) << "Out of Range error: " << oor.what() << " when parse " << input_str;
    return false;
  } catch (const std::invalid_argument &ia) {
    MS_LOG(ERROR) << "Invalid argument: " << ia.what() << " when parse " << input_str;
    return false;
  }
  return true;
}

void CheckStringMatch(size_t start, size_t end, std::string *matched_str, const std::string &input_str) {
  if (start != std::string::npos && end != std::string::npos && end > start && start + 1 < input_str.length()) {
    *matched_str = input_str.substr(start + 1, end - (start + 1));
  }
}
}  // namespace mindspore
