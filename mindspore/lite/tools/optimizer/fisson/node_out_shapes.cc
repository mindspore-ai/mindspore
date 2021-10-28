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

#include "tools/optimizer/fisson/node_out_shapes.h"
#include <vector>
#include "tools/optimizer/parallel/spliter.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
AnfNodePtr NodeOutShapes::Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, nullptr, "input func_graph is nullptr");
  MS_CHECK_TRUE_MSG(node != nullptr, nullptr, "input node is nullptr");
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  std::vector<ShapeVector> input_shapes;
  std::vector<ShapeVector> output_shapes;
  auto cnode = node->cast<CNodePtr>();
  // assume multi inputs
  for (const auto &input_node : cnode->inputs()) {
    MS_ASSERT(input_node != nullptr);
    if (utils::isa<CNodePtr>(input_node) || utils::isa<ParameterPtr>(input_node)) {
      auto in_shape = input_node->Shape();
      if (in_shape == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return nullptr;
      }
      if (utils::isa<abstract::ShapePtr>(in_shape)) {
        const auto &shape = in_shape->cast<abstract::ShapePtr>()->shape();
        input_shapes.push_back(shape);
      }
    }
  }
  // assume multi outputs
  auto out_shape = cnode->Shape();
  if (out_shape == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }
  if (utils::isa<abstract::TupleShapePtr>(out_shape)) {
    auto shape = out_shape->cast<abstract::TupleShapePtr>();
    for (size_t i = 0; i < shape->size(); ++i) {
      const auto &shape_ptr = (*shape)[i];
      if (!utils::isa<abstract::ShapePtr>(shape_ptr)) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return nullptr;
      }
      output_shapes.push_back(shape_ptr->cast<abstract::ShapePtr>()->shape());
    }
  } else if (utils::isa<abstract::ShapePtr>(out_shape)) {
    const auto &shape = out_shape->cast<abstract::ShapePtr>()->shape();
    output_shapes.push_back(shape);
  }
  std::string node_name = cnode->fullname_with_scope();
  Spliter::GetInstance()->UpdateNodeInputShapes(node_name, input_shapes);
  Spliter::GetInstance()->UpdateNodeOutputShapes(node_name, output_shapes);
  return node;
}
}  // namespace opt
}  // namespace mindspore
