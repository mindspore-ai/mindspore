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
#include "src/litert/tensor_category.h"
#include "src/common/utils.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
Category TensorCategory(const int node_type, const size_t shape_num, const TypeId data_type, const size_t data_size) {
  return (node_type == NodeType_ValueNode)
           ? (shape_num == 0 && data_size == DataTypeSize(data_type) ? Category::CONST_SCALAR : Category::CONST_TENSOR)
           : Category::VAR;
}

Category TensorCategory(const schema::Tensor &tensor) {
  auto shape_num = tensor.dims() == nullptr ? 0 : tensor.dims()->size();
  auto data_size = tensor.data() == nullptr ? 0 : tensor.data()->size();
  return TensorCategory(tensor.nodeType(), shape_num, TypeId(tensor.dataType()), data_size);
}
}  // namespace lite
}  // namespace mindspore
