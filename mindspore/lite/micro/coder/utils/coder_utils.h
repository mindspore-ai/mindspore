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
#ifndef MINDSPORE_LITE_MICRO_CODER_UTILS_CODER_UTILS_H_
#define MINDSPORE_LITE_MICRO_CODER_UTILS_CODER_UTILS_H_

#include <set>
#include <limits>
#include <vector>
#include <memory>
#include <string>
#include "include/errorcode.h"
#include "securec/include/securec.h"
#include "src/tensor.h"
#include "coder/opcoders/op_coder.h"

namespace mindspore::lite::micro {

constexpr int kWeightPrecision = 9;

std::vector<std::string> AddDumpDataInfo(const std::vector<std::string> &blocks,
                                         const std::vector<std::unique_ptr<OperatorCoder>> &opcoders);

void PrintTensorData(const lite::Tensor *tensor, std::ofstream &ofs);

template <typename T>
std::string ArrayToString(std::vector<T> array) {
  std::string result;
  std::for_each(array.begin(), array.end(), [&result](const T &t) { result += std::to_string(t) + ", "; });
  return "{" + result + "}";
}

}  // namespace mindspore::lite::micro

#endif  // MINDSPORE_LITE_MICRO_CODER_UTILS_CODER_UTILS_H_
