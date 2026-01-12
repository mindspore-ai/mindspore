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
#include "backend/ge_backend/pass/promote_cast_for_binary_ops.h"
#include <memory>
#include <vector>
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_f.h"
#include "ops_utils/op_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace opt {
const BaseRef PromoteCastForBinaryOp::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr W = std::make_shared<Var>();
  return VectorRef({prim_, V, W});
}

namespace {
AnfNodePtr CreateCastNode(const FuncGraphPtr &graph, const AnfNodePtr &input, const TypeId dst_type) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  if (common::AnfAlgo::GetOutputInferDataType(input, 0) != dst_type) {
    AnfNodePtr cast = graph->NewCNode({NewValueNode(std::make_shared<Primitive>("Cast")), input});
    MS_EXCEPTION_IF_NULL(cast);
    common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {AnfAlgo::GetOutputDetailShape(input, 0)}, cast.get());
    common::AnfAlgo::SetNodeAttr(kAttrDstType, TypeIdToType(dst_type), cast);
    cast->set_scope(input->scope());
    return cast;
  }
  return input;
}
}  // namespace

const AnfNodePtr PromoteCastForBinaryOp::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  constexpr size_t kValidBinaryNodeInputNum = 3;
  if (cnode->size() != kValidBinaryNodeInputNum) {
    MS_LOG(INFO) << "The input number of " << prim_->name() << " should be 2, but got " << (cnode->size() - 1);
    return nullptr;
  }

  auto in0 = cnode->input(1);
  auto in1 = cnode->input(2);
  auto type0 = common::AnfAlgo::GetOutputInferDataType(in0, 0);
  auto type1 = common::AnfAlgo::GetOutputInferDataType(in1, 0);
  if (type0 == type1) {
    return nullptr;
  }

  TypeId target_type = ops::PromoteType(type0, type1, prim_->name());
  AnfNodePtr new_in0 = in0;
  AnfNodePtr new_in1 = in1;

  if (type0 != target_type) {
    new_in0 = CreateCastNode(graph, in0, target_type);
  }
  if (type1 != target_type) {
    new_in1 = CreateCastNode(graph, in1, target_type);
  }

  std::vector<AnfNodePtr> new_inputs = {cnode->input(0), new_in0, new_in1};
  auto new_node = NewCNode(new_inputs, graph);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_scope(node->scope());
  new_node->set_fullname_with_scope(node->fullname_with_scope());

  // Set abstract for the new node with target_type
  common::AnfAlgo::SetOutputTypeAndDetailShape({target_type}, {AnfAlgo::GetOutputDetailShape(node, 0)}, new_node.get());

  // If the target type is different from the original output type, cast it back
  auto origin_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (target_type != origin_type) {
    MS_LOG(INFO) << "PromoteCastForBinaryOp: " << prim_->name() << " promoted to " << TypeIdLabel(target_type)
                 << ", casting back to " << TypeIdLabel(origin_type);
    return CreateCastNode(graph, new_node, origin_type);
  }
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
