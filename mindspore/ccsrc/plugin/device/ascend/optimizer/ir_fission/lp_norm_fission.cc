/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/lp_norm_fission.h"
#include <vector>
#include <memory>
#include "ops/math_op_name.h"
#include "ops/math_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kLpNormRealInputNum = 1;
}  // namespace

AnfNodePtr LpNormFission::CreateLpNormReduceV2(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                               const AnfNodePtr &cast_node) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(anf_node);
  auto lp_cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(lp_cnode);
  if (lp_cnode->inputs().size() < kLpNormRealInputNum + 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "The input size of node " + lp_cnode->DebugString() + " is less than "
                               << (kLpNormRealInputNum + 1) << trace::DumpSourceLines(anf_node);
  }
  std::vector<AnfNodePtr> lp_norm_reduce_v2_inputs;
  if (cast_node == nullptr) {
    lp_norm_reduce_v2_inputs = {NewValueNode(std::make_shared<Primitive>(kLpNormReduceV2OpName)), lp_cnode->input(1)};
  } else {
    lp_norm_reduce_v2_inputs = {NewValueNode(std::make_shared<Primitive>(kLpNormReduceV2OpName)), cast_node};
  }
  auto lp_norm_reduce_v2 = NewCNode(lp_norm_reduce_v2_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(lp_norm_reduce_v2);

  lp_norm_reduce_v2->set_abstract(anf_node->abstract());
  lp_norm_reduce_v2->set_scope(anf_node->scope());

  std::vector<TypeId> dtypes = {kNumberTypeFloat32};
  std::vector<ShapeVector> shapes = {common::AnfAlgo::GetOutputInferShape(anf_node, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, lp_norm_reduce_v2.get());

  auto get_p = common::AnfAlgo::GetNodeAttr<int64_t>(anf_node, "p");
  common::AnfAlgo::SetNodeAttr("p", MakeValue(static_cast<float>(get_p)), lp_norm_reduce_v2);
  common::AnfAlgo::CopyNodeAttr("axis", lp_cnode, lp_norm_reduce_v2);
  common::AnfAlgo::CopyNodeAttr("keep_dims", lp_cnode, lp_norm_reduce_v2);
  common::AnfAlgo::CopyNodeAttr("epsilon", lp_cnode, lp_norm_reduce_v2);
  return lp_norm_reduce_v2;
}

AnfNodePtr LpNormFission::CreateLpNormUpdateV2(const FuncGraphPtr &func_graph, const AnfNodePtr &anf_node,
                                               const AnfNodePtr &lp_norm_reduce_v2_outputs) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(anf_node);
  auto lp_cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(lp_cnode);
  if (lp_cnode->inputs().size() < kLpNormRealInputNum + 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "The input size of node " + lp_cnode->DebugString() + " is less than "
                               << (kLpNormRealInputNum + 1) << trace::DumpSourceLines(anf_node);
  }

  std::vector<AnfNodePtr> lp_norm_update_v2_inputs = {NewValueNode(std::make_shared<Primitive>(kLpNormUpdateV2OpName)),
                                                      lp_norm_reduce_v2_outputs};
  auto lp_norm_update_v2 = NewCNode(lp_norm_update_v2_inputs, func_graph);
  MS_EXCEPTION_IF_NULL(lp_norm_update_v2);

  lp_norm_update_v2->set_abstract(anf_node->abstract());
  lp_norm_update_v2->set_scope(anf_node->scope());

  std::vector<TypeId> dtypes = {kNumberTypeFloat32};
  std::vector<ShapeVector> shapes = {common::AnfAlgo::GetOutputInferShape(anf_node, 0)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, lp_norm_update_v2.get());

  auto get_p = common::AnfAlgo::GetNodeAttr<int64_t>(anf_node, "p");
  common::AnfAlgo::SetNodeAttr("p", MakeValue(static_cast<float>(get_p)), lp_norm_update_v2);
  common::AnfAlgo::CopyNodeAttr("epsilon", lp_cnode, lp_norm_update_v2);
  return lp_norm_update_v2;
}

const BaseRef LpNormFission::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

AnfNodePtr CreateCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input, const TypeId dst_type, bool is_first) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  AnfNodePtr cast_node = nullptr;
  if (is_first) {
    auto lp_cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(lp_cnode);
    cast_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kCastOpName)), lp_cnode->input(1)});

    auto input_shape = lp_cnode->input(1)->Shape();
    MS_EXCEPTION_IF_NULL(input_shape);
    auto shape_ptr = input_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);

    ShapeVector input_shape_lp = shape_ptr->shape();
    std::vector<ShapeVector> shapes_final;
    shapes_final.push_back(input_shape_lp);
    common::AnfAlgo::SetOutputInferTypeAndShape({dst_type}, shapes_final, cast_node.get());
  } else {
    cast_node = graph->NewCNode({NewValueNode(std::make_shared<Primitive>(kCastOpName)), input});
    common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {AnfAlgo::GetOutputDetailShape(input, 0)},
                                                 cast_node.get());
  }
  MS_EXCEPTION_IF_NULL(cast_node);

  common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(dst_type), cast_node);
  cast_node->set_scope(input->scope());
  return cast_node;
}

const AnfNodePtr LpNormFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  if (op_name != kLpNormOpName) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->size() < kLpNormRealInputNum + 1) {
    MS_LOG(INFO) << "The input num of BatchNorm less than" << kLpNormRealInputNum << ". The node should not be changed";
    return nullptr;
  }

  AnfNodePtr begin_cast_node = nullptr;
  bool need_cast = False;

  if (common::AnfAlgo::GetOutputInferDataType(node, 0) != kNumberTypeFloat32) {
    begin_cast_node = CreateCastNode(func_graph, node, kNumberTypeFloat32, true);
    need_cast = True;
  }
  AnfNodePtr lp_norm_reduce_v2_outputs = CreateLpNormReduceV2(func_graph, node, begin_cast_node);
  AnfNodePtr lp_norm_update_v2_outputs = CreateLpNormUpdateV2(func_graph, node, lp_norm_reduce_v2_outputs);
  if (need_cast) {
    AnfNodePtr end_cast_node =
      CreateCastNode(func_graph, lp_norm_update_v2_outputs, common::AnfAlgo::GetOutputInferDataType(node, 0), false);
    return end_cast_node;
  }
  return lp_norm_update_v2_outputs;
}
}  // namespace opt
}  // namespace mindspore
