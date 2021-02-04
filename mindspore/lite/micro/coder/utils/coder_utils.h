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
#ifndef MICRO_CODER_CODER_UTILS_CODER_UTILS_H_
#define MICRO_CODER_CODER_UTILS_CODER_UTILS_H_

#include <limits>
#include <vector>
#include <string>
#include "include/errorcode.h"
#include "securec/include/securec.h"
#include "src/tensor.h"

namespace mindspore::lite::micro {

constexpr int kSubSize = 2;
constexpr int kDefaultDims = 4;

std::string EnumNameDataType(TypeId type);

}  // namespace mindspore::lite::micro

#endif  // MICRO_CODER_CODER_UTILS_CODER_UTILS_H_
