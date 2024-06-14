/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_SIMPLE_INFO_H_
#define MINDSPORE_CORE_UTILS_SIMPLE_INFO_H_

#include <memory>
#include <vector>
#include "mindapi/base/shape_vector.h"
#include "ir/dtype/type.h"

namespace mindspore {
// Save the shape, dtype and object type info of op inputs or outputs, where size is the size of the inputs or outputs
// element
struct ValueSimpleInfo {
  // Is tuple output
  bool is_tuple_output_{false};
  // Number of total element
  size_t size_;
  // ShapeVector of every element
  ShapeArray shape_vector_;
  // TypePtr of every element
  TypePtrList dtype_vector_;
  // Object type of every element
  std::vector<TypeId> object_type_vector_;
};
using ValueSimpleInfoPtr = std::shared_ptr<ValueSimpleInfo>;
}  // namespace mindspore
#endif
