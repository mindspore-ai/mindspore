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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_TENSOR_CATEGORY_H_
#define MINDSPORE_LITE_SRC_RUNTIME_TENSOR_CATEGORY_H_

#include <cstddef>
#include "ir/dtype/type_id.h"

namespace mindspore {
namespace schema {
struct Tensor;
}
namespace lite {
enum Category {
  VAR,           // activation tensor
  CONST_TENSOR,  // weight tensor
  CONST_SCALAR,  // weight scalar
  GRAPH_INPUT,
  GRAPH_OUTPUT,
  PARAMETER,
};

Category TensorCategory(const int node_type, const size_t shape_num, const TypeId data_type, const size_t data_size);
Category TensorCategory(const schema::Tensor &tensor);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_TENSOR_CATEGORY_H_
