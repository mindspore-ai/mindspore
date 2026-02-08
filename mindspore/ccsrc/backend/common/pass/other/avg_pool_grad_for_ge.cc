/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/other/avg_pool_grad_for_ge.h"

#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "ir/tensor_new.h"
#include "primitive/conv_pool_ops.h"
#include "primitive/array_ops.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/pass_manager/helper.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace opt {
std::vector<std::string> AvgPoolGradForGE::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimAvgPoolGrad->name()};
  return ret;
}

const BaseRef AvgPoolGradForGE::DefinePattern() const {
  auto x = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimAvgPoolGrad, x});
}

const AnfNodePtr AvgPoolGradForGE::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  // AvgPoolGrad(x, out, dout, ....) ===> AvgPoolGrad(x_shape, dout, ....)
  MS_EXCEPTION_IF_NULL(graph);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto avg_pool_grad_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(avg_pool_grad_node);
  // get shape node
  auto input_x = avg_pool_grad_node->input(kIndex1);
  MS_EXCEPTION_IF_NULL(input_x);
  auto input_x_shape = input_x->Shape();
  AnfNodePtr shape_node = nullptr;
  MS_EXCEPTION_IF_NULL(input_x_shape);
  if (input_x_shape->IsDynamic()) {
    shape_node = CreateTensorShapeNode(graph, input_x, node);
  } else {
    MS_EXCEPTION_IF_NULL(input_x_shape->cast<abstract::ShapePtr>());
    auto shape_vector = input_x_shape->cast<abstract::ShapePtr>()->shape();
    std::vector<int32_t> value_node_data;
    (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(value_node_data), LongToInt);
    auto shape_tensor = tensor::from_vector(value_node_data, TypeIdToType(TypeId::kNumberTypeInt32));
    shape_node = AnfAlgo::ConvertValueToNode(kernel_graph, shape_tensor);
  }
  // new avg_pool_grad_node
  auto inputs = avg_pool_grad_node->inputs();
  auto op_node = inputs.at(0);
  std::vector<AnfNodePtr> new_inputs = {op_node, shape_node};
  // Reset ValueDepend
  MS_EXCEPTION_IF_NULL(op_node);
  auto prim = GetValueNode<PrimitivePtr>(op_node);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<int64_t> depend_indices{0, 2, 3, 4};
  prim->AddAttr(kAttrValueDepend, MakeValue(depend_indices));
  constexpr size_t dout_index = kIndex3;
  for (size_t i = dout_index; i < inputs.size(); ++i) {
    new_inputs.push_back(inputs[i]);
  }
  auto new_node = kernel_graph->NewCNodeWithInfos(new_inputs, avg_pool_grad_node);
  new_node->set_abstract(avg_pool_grad_node->abstract());
  new_node->set_inputs(new_inputs);
  new_node->set_scope(avg_pool_grad_node->scope());
  new_node->set_fullname_with_scope(avg_pool_grad_node->fullname_with_scope());
  return new_node;
}

CNodePtr AvgPoolGradForGE::CreateTensorShapeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const AnfNodePtr &grad_node) const {
  MS_EXCEPTION_IF_NULL(grad_node);
  auto prim = std::make_shared<Primitive>(kTensorShapeOpName);
  MS_EXCEPTION_IF_NULL(prim);
  AnfNodePtrList inputs = {NewValueNode(prim), node};
  CNodePtr tensor_shape_node = func_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tensor_shape_node);
  tensor_shape_node->set_scope(grad_node->scope());
  auto abs = InferAbstract(prim, {node});
  MS_EXCEPTION_IF_NULL(abs);
  tensor_shape_node->set_abstract(abs);
  // cast int64 -> int32
  auto cast_to_node = AddCastNode(func_graph, kNumberTypeInt32, tensor_shape_node, false);
  return cast_to_node;
}
}  // namespace opt
}  // namespace mindspore
