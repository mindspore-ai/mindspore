/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_COMMON_STRING_UTIL_H_
#define MINDSPORE_LITE_SRC_COMMON_STRING_UTIL_H_
#include <vector>
#include <string>
#include <utility>
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "tools/common/option.h"
#include "include/errorcode.h"
#include "include/lite_utils.h"

namespace mindspore {
namespace lite {
typedef struct StringPack {
  int len = 0;
  const char *data = nullptr;
} StringPack;

// example of string tensor:
// 2, 0, 0, 0    # int32, num of strings
// 16, 0, 0, 0   # int32, offset of 0-th string
// 21, 0, 0, 0   # int32, offset of 1-th string
// 30, 0, 0, 0   # int32, total length of tensor data
// 'h', 'e', 'l', 'l', 'o', 'h', 'o', 'w', 'a', 'r', 'e', 'y', 'o', 'u'  # char, "hello", "how are you"
std::vector<StringPack> ParseTensorBuffer(Tensor *tensor);
std::vector<StringPack> ParseStringBuffer(const void *data);

int WriteStringsToTensor(Tensor *tensor, const std::vector<StringPack> &string_buffer);
int WriteSeperatedStringsToTensor(Tensor *tensor, const std::vector<std::vector<StringPack>> &string_buffer);

int GetStringCount(const void *data);
int GetStringCount(Tensor *tensor);
uint64_t StringHash64(const char *s, size_t len);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_COMMON_STRING_UTIL_H_
