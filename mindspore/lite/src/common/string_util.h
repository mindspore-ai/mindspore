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

#ifndef MINDSPORE_LITE_COMMON_STRING_UTIL_H_
#define MINDSPORE_LITE_COMMON_STRING_UTIL_H_

#include <vector>
#include <string>
#include <utility>
#include "mindspore/lite/src/tensor.h"
#include "src/common/log_adapter.h"
#include "tools/common/option.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
typedef struct {
  int len;
  const char *data;
} StringPack;

std::vector<StringPack> ParseTensorBuffer(Tensor *tensor);
std::vector<StringPack> ParseStringBuffer(const void *data);

int WriteStringsToTensor(Tensor *tensor, const std::vector<StringPack> &string_buffer);
int WriteSeperatedStringsToTensor(Tensor *tensor, const std::vector<std::vector<StringPack>> &string_buffer);

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_COMMON_STRING_UTIL_H_
