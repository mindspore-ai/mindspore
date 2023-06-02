/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/maximum_grad_fission.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/backend/kernel_graph.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "plugin/device/ascend/optimizer/create_node_helper.h"
#include "utils/trace_base.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kMaximumGradInputNum = 4;
AnfNodePtr CreateCNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &inputs,
                       const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(std::make_shared<Primitive>(op_name))};
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    (void)new_node_inputs.emplace_back(input);
  }
  return NewCNode(new_node_inputs, func_graph);
}
AnfNodePtr CreateTensorOutputCNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &inputs,
                                   const std::string &op_name, const ShapeVector &shape_vector, const TypeId &type_id) {
  auto cnode = CreateCNode(func_graph, inputs, op_name);
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), shape_vector);
  cnode->set_abstract(abs);
  return cnode;
}
AnfNodePtr CreatTupleOutputCNode(const FuncGraphPtr &func_graph, const std::vector<AnfNodePtr> &inputs,
                                 const std::string &op_name, const std::vector<ShapeVector> &shape_vectors,
                                 const std::vector<TypeId> &type_ids) {
  if (shape_vectors.size() != type_ids.size()) {
    return nullptr;
  }
  auto cnode = CreateCNode(func_graph, inputs, op_name);
  std::vector<abstract::AbstractBasePtr> abstract_list;
  for (size_t i = 0; i < shape_vectors.size(); ++i) {
    (void)abstract_list.emplace_back(
      std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_ids[i]), shape_vectors[i]));
  }
  cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return cnode;
}

AnfNodePtr CreateValueNodeFromAxis(size_t axis_num) {
  std::vector<int64_t> axis;
  for (size_t i = 0; i < axis_num; ++i) {
    (void)axis.emplace_back(SizeToLong(i));
  }
  auto tensor = std::make_shared<tensor::Tensor>(axis, TypeIdToType(TypeId::kNumberTypeInt64));
  MS_EXCEPTION_IF_NULL(tensor);
  auto value_node = NewValueNode(tensor);
  ShapeVector shape = {SizeToLong(axis_num)};
  auto abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(TypeId::kNumberTypeInt64), shape);
  value_node->set_abstract(abs);
  return value_node;
}
}  // namespace

const BaseRef MaximumGradFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimMaximumGrad, Xs});
}

const AnfNodePtr MaximumGradFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto maximum_grad_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(maximum_grad_cnode);
  if (maximum_grad_cnode->inputs().size() < kMaximumGradInputNum) {
    MS_LOG(WARNING) << "Invalid maximum node:" << maximum_grad_cnode->DebugString();
    return maximum_grad_cnode;
  }
  const auto &input_x = maximum_grad_cnode->input(1);
  const auto &input_y = maximum_grad_cnode->input(2);
  const auto &input_dout = maximum_grad_cnode->input(3);
  MS_EXCEPTION_IF_NULL(input_x);
  MS_EXCEPTION_IF_NULL(input_y);
  MS_EXCEPTION_IF_NULL(input_dout);
  if (((!IsDynamicShape(common::AnfAlgo::GetOutputInferShape(input_x, 0))) &&
       !IsDynamicShape(common::AnfAlgo::GetOutputInferShape(input_y, 0))) ||
      ((!common::AnfAlgo::GetOutputInferShape(input_x, 0).empty()) &&
       (!common::AnfAlgo::GetOutputInferShape(input_y, 0).empty()))) {
    return maximum_grad_cnode;
  }
  auto zeroslike_node = CreateTensorOutputCNode(func_graph, {input_dout}, prim::kPrimZerosLike->name(),
                                                common::AnfAlgo::GetOutputInferShape(input_dout, 0),
                                                common::AnfAlgo::GetOutputInferDataType(input_dout, 0));
  const auto &greaterequal_shape =
    (IsDynamic(common::AnfAlgo::GetOutputInferShape(input_x, 0)) ? common::AnfAlgo::GetOutputInferShape(input_x, 0)
                                                                 : common::AnfAlgo::GetOutputInferShape(input_y, 0));
  auto greateequal_node = CreateTensorOutputCNode(func_graph, {input_x, input_y}, prim::kPrimGreaterEqual->name(),
                                                  greaterequal_shape, TypeId::kNumberTypeBool);
  auto select_dx_node = CreateTensorOutputCNode(
    func_graph, {greateequal_node, input_dout, zeroslike_node}, prim::kPrimSelect->name(),
    common::AnfAlgo::GetOutputInferShape(input_dout, 0), common::AnfAlgo::GetOutputInferDataType(input_dout, 0));
  auto select_dy_node = CreateTensorOutputCNode(
    func_graph, {greateequal_node, zeroslike_node, input_dout}, prim::kPrimSelect->name(),
    common::AnfAlgo::GetOutputInferShape(input_dout, 0), common::AnfAlgo::GetOutputInferDataType(input_dout, 0));
  auto dx_node = select_dx_node;
  auto dy_node = select_dy_node;
  if (common::AnfAlgo::GetOutputInferShape(input_x, 0).empty()) {
    auto axis_valuenode = CreateValueNodeFromAxis(greaterequal_shape.size());
    dx_node = CreateTensorOutputCNode(func_graph, {select_dx_node, axis_valuenode}, prim::kPrimReduceSum->name(), {},
                                      common::AnfAlgo::GetOutputInferDataType(input_x, 0));
  }
  if (common::AnfAlgo::GetOutputInferShape(input_y, 0).empty()) {
    auto axis_valuenode = CreateValueNodeFromAxis(greaterequal_shape.size());
    dy_node = CreateTensorOutputCNode(func_graph, {select_dy_node, axis_valuenode}, prim::kPrimReduceSum->name(), {},
                                      common::AnfAlgo::GetOutputInferDataType(input_y, 0));
  }
  auto maketuple_node = CreatTupleOutputCNode(
    func_graph, {dx_node, dy_node}, prim::kPrimMakeTuple->name(),
    {common::AnfAlgo::GetOutputInferShape(dx_node, 0), common::AnfAlgo::GetOutputInferShape(dy_node, 0)},
    {common::AnfAlgo::GetOutputInferDataType(dx_node, 0), common::AnfAlgo::GetOutputInferDataType(dy_node, 0)});
  return maketuple_node == nullptr ? maximum_grad_cnode : maketuple_node;
}
}  // namespace opt
}  // namespace mindspore
