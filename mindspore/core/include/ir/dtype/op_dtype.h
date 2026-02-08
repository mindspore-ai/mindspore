/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <cstdint>

#ifndef MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DTYPE_OP_DTYPE_H_
#define MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DTYPE_OP_DTYPE_H_
namespace mindspore::ops {
enum OP_DTYPE : int64_t {
  DT_BEGIN = 0,
  DT_BOOL,
  DT_INT,
  DT_FLOAT,
  DT_NUMBER,
  DT_TENSOR,
  DT_STR,
  DT_ANY,
  DT_TUPLE_BOOL,
  DT_TUPLE_INT,
  DT_TUPLE_FLOAT,
  DT_TUPLE_NUMBER,
  DT_TUPLE_TENSOR,
  DT_TUPLE_STR,
  DT_TUPLE_ANY,
  DT_LIST_BOOL,
  DT_LIST_INT,
  DT_LIST_FLOAT,
  DT_LIST_NUMBER,
  DT_LIST_TENSOR,
  DT_LIST_STR,
  DT_LIST_ANY,
  DT_TYPE,
  DT_NONE,
  DT_STORAGE,
  DT_END,
};
}  // namespace mindspore::ops
#endif  // MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DTYPE_OP_DTYPE_H_
