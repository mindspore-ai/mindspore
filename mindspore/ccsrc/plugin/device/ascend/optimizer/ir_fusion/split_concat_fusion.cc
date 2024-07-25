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

#include "plugin/device/ascend/optimizer/ir_fusion/split_concat_fusion.h"

#include <string>
#include <vector>

#include "mindspore/core/ops/nn_ops.h"

#include "abstract/dshape.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace opt {
const BaseRef SplitConcatFusion::DefinePattern() const {
  split_axis_ = std::make_shared<CondVar>(IsConstant);
  concat_axis_ = std::make_shared<CondVar>(IsConstant);
  output_num_ = std::make_shared<CondVar>(IsConstant);
  VectorRef split_ops = VectorRef({prim::kPrimSplit, x1_, split_axis_, output_num_});
  global_rank_size_ = distributed::collective::CollectiveManager::instance()->global_rank_size();
  std::vector<BaseRef> tuple_elements;
  tuple_elements.emplace_back(prim::kPrimMakeTuple);
  for (size_t i = 0; i < global_rank_size_; i++) {
    VarPtr index = std::make_shared<CondVar>(IsConstant);
    VectorRef tuple_get_item = VectorRef({prim::kPrimTupleGetItem, split_ops, index});
    tuple_elements.emplace_back(tuple_get_item);
  }
  auto tuple_node = VectorRef(tuple_elements);
  return VectorRef({prim::kPrimConcat, tuple_node, concat_axis_});
}

const AnfNodePtr SplitConcatFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  auto env = common::GetEnv("MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST");
  if (env.find("SplitConcatFusion") != std::string::npos) {
    return nullptr;
  }

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(equiv);
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 0);
  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto split_axis = GetValue<int64_t>(GetValueNode(utils::cast<AnfNodePtr>((*equiv)[split_axis_])));
  auto concat_axis = GetValue<int64_t>(GetValueNode(utils::cast<AnfNodePtr>((*equiv)[concat_axis_])));
  auto output_num = GetValue<int64_t>(GetValueNode(utils::cast<AnfNodePtr>((*equiv)[output_num_])));
  constexpr size_t expected_shape_size = 2;
  constexpr int64_t dynamic_shape = -1;
  auto concat_size = shape[1];

  if (split_axis != 0 || concat_axis != 1 || output_num != global_rank_size_ || shape.size() != expected_shape_size ||
      shape[0] != dynamic_shape || concat_size == dynamic_shape) {
    MS_LOG(INFO) << "split_axis: " << split_axis << " concat_axis: " << concat_axis << " output_num:" << output_num
                 << " shape:" << shape << " is unexpected.";
    return nullptr;
  }

  ShapeVector target_shape1 = {output_num, -1, concat_size};
  ShapeVector target_shape2 = {-1, concat_size * output_num};
  auto output_type = common::AnfAlgo::GetOutputInferDataType(node, 0);

  auto reshape1_prim = std::make_shared<Primitive>("Reshape");
  std::vector<AnfNodePtr> inputs = {NewValueNode(reshape1_prim), x1, CreateShapeValueNode(graph, target_shape1, false)};
  auto first_reshape = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(first_reshape);
  std::vector<TypeId> first_reshape_types;
  std::vector<BaseShapePtr> first_reshape_shapes;
  first_reshape_types.push_back(output_type);
  first_reshape_shapes.push_back(std::make_shared<abstract::TensorShape>(target_shape1));
  common::AnfAlgo::SetOutputTypeAndDetailShape(first_reshape_types, first_reshape_shapes, first_reshape.get());
  first_reshape->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(first_reshape);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, first_reshape.get());

  ShapeVector target_axis = {1, 0, 2};
  auto transpose_prim = std::make_shared<Primitive>(prim::kPrimTranspose->name());
  inputs = {NewValueNode(transpose_prim), first_reshape, CreateShapeValueNode(graph, target_axis, false)};
  auto tranpose_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(tranpose_node);
  std::vector<TypeId> tranpose_types;
  std::vector<BaseShapePtr> tranpose_shapes;
  tranpose_types.push_back(output_type);
  tranpose_shapes.push_back(std::make_shared<abstract::TensorShape>(target_axis));
  common::AnfAlgo::SetOutputTypeAndDetailShape(tranpose_types, tranpose_shapes, tranpose_node.get());
  tranpose_node->set_scope(node->scope());
  build_info = GenerateKernelBuildInfo(tranpose_node);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, tranpose_node.get());

  auto reshape2_prim = std::make_shared<Primitive>("Reshape");
  inputs = {NewValueNode(reshape2_prim), tranpose_node, CreateShapeValueNode(graph, target_shape2, false)};
  auto second_reshape = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(second_reshape);
  std::vector<TypeId> second_reshape_types;
  std::vector<BaseShapePtr> second_reshape_shapes;
  second_reshape_types.push_back(output_type);
  second_reshape_shapes.push_back(std::make_shared<abstract::TensorShape>(target_shape2));
  common::AnfAlgo::SetOutputTypeAndDetailShape(second_reshape_types, second_reshape_shapes, second_reshape.get());
  second_reshape->set_scope(node->scope());

  build_info = GenerateKernelBuildInfo(second_reshape);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, second_reshape.get());
  return second_reshape;
}
}  // namespace opt
}  // namespace mindspore
