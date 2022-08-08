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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_SHAPE_VECTOR_H_
#define MINDSPORE_CORE_MINDAPI_BASE_SHAPE_VECTOR_H_

#include <cstdint>
#include <vector>

using ShapeValueDType = int64_t;
using ShapeVector = std::vector<ShapeValueDType>;
using ShapeArray = std::vector<ShapeVector>;

#endif  // MINDSPORE_CORE_MINDAPI_BASE_SHAPE_VECTOR_H_
