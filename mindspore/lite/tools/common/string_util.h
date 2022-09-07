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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_STRING_UTIL_H_
#define MINDSPORE_LITE_TOOLS_COMMON_STRING_UTIL_H_
#include <vector>
#include <string>
#include <utility>
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "tools/common/option.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
bool EraseBlankSpaceAndLineBreak(std::string *input_string);

bool EraseQuotes(std::string *input_string);

bool FindAndReplaceAll(std::string *input_str, const std::string &search, const std::string &replace);

MS_API std::vector<std::string> SplitStringToVector(const std::string &raw_str, const char &delimiter);

std::vector<std::string> SplitStringToVector(const std::string &raw_str, const std::string &delimiter);

MS_API bool ConvertIntNum(const std::string &str, int *value);

bool ConvertDoubleNum(const std::string &str, double *value);

bool ConvertBool(std::string str, bool *value);

bool ConvertDoubleVector(const std::string &str, std::vector<double> *value);

size_t Hex2ByteArray(const std::string &hex_str, unsigned char *byte_array, size_t max_len);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_COMMON_STRING_UTIL_H_
