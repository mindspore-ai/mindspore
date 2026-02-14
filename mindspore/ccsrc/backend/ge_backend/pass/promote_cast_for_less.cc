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
#include "backend/ge_backend/pass/promote_cast_for_less.h"
#include <memory>
#include <vector>
#include "include/utils/anfalgo.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "ops_utils/op_utils.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace opt {
const BaseRef PromoteCastForLess::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr W = std::make_shared<Var>();
  return VectorRef({std::make_shared<Primitive>(prim::kPrimLess->name()), V, W});
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

const AnfNodePtr PromoteCastForLess::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                             const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  constexpr size_t kValidLessNodeInputNum = 3;
  if (cnode->size() != kValidLessNodeInputNum) {
    MS_LOG(EXCEPTION) << "The input number of Less should be 2, but got " << (cnode->size() - 1);
    return nullptr;
  }

  auto in0 = cnode->input(1);
  auto in1 = cnode->input(2);
  auto type0 = common::AnfAlgo::GetOutputInferDataType(in0, 0);
  auto type1 = common::AnfAlgo::GetOutputInferDataType(in1, 0);

  if (type0 == type1 && type0 != kNumberTypeBool) {
    return nullptr;
  }

  TypeId target_type = ops::PromoteType(type0, type1, "Less");
  // For Less, if the target type is bool (e.g., bool < bool), we promote it to int32 to avoid
  // compatibility issues with some backends that do not support direct boolean comparison.
  if (target_type == kNumberTypeBool) {
    target_type = kNumberTypeInt32;
  }

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
  new_node->set_abstract(node->abstract());
  new_node->set_scope(node->scope());
  new_node->set_fullname_with_scope(node->fullname_with_scope());
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
