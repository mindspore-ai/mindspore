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

#ifndef MINDSPORE_LITE_INTERNAL_INCLUDE_LITE_UTILS_H_
#define MINDSPORE_LITE_INTERNAL_INCLUDE_LITE_UTILS_H_
#include <vector>
#include <string>

struct MSTensor;
struct Node;
using TensorPtr = MSTensor *;
using TensorPtrVector = std::vector<MSTensor *>;
using Uint32Vector = std::vector<uint32_t>;
using String = std::string;
using StringVector = std::vector<std::string>;
using ShapeVector = std::vector<int>;
using NodePtrVector = std::vector<struct Node *>;
using Int32Vector = std::vector<int>;
using Int32VectorVector = std::vector<Int32Vector>;
#endif  // MINDSPORE_LITE_INCLUDE_LITE_UTILS_H_
