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
#include "tools/optimizer/fusion/glu_fusion.h"
#include <memory>
#include <string>
#include "ops/glu.h"
#include "utils/utils.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
CNodePtr GLUFusion::CreateGLUNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  auto glu_prim = std::make_shared<ops::GLU>();
  MS_CHECK_TRUE_RET(glu_prim != nullptr, nullptr);
  auto split_prim = GetValueNode<PrimitivePtr>(utils::cast<AnfNodePtr>((*equiv)[split_prim_]));
  if (split_prim != nullptr && split_prim->GetAttr(ops::kAxis) != nullptr) {
    auto axis = GetValue<int64_t>(split_prim->GetAttr(ops::kAxis));
    glu_prim->set_axis(axis);
  }
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  auto glu_cnode = func_graph->NewCNode(glu_prim, {input_node});
  MS_CHECK_TRUE_RET(glu_cnode != nullptr, nullptr);
  glu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_glu");
  if (node->abstract() != nullptr) {
    glu_cnode->set_abstract(node->abstract()->Clone());
  }
  return glu_cnode;
}

bool GLUFusion::Init() const {
  input_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_ != nullptr, false);
  axis_ = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(axis_ != nullptr, false);
  split_prim_ = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSplit>);
  MS_CHECK_TRUE_RET(split_prim_ != nullptr, false);
  return true;
}

const BaseRef GLUFusion::DefinePattern() const {
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return {};
  }
  VectorRef split_ref({split_prim_, input_, axis_});
  auto is_tuple_getitem1 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>);
  MS_CHECK_TRUE_RET(is_tuple_getitem1 != nullptr, {});
  auto is_seq_var1 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var1 != nullptr, {});
  VectorRef tuple_ref1({is_tuple_getitem1, split_ref, is_seq_var1});
  auto is_tuple_getitem2 = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>);
  MS_CHECK_TRUE_RET(is_tuple_getitem2 != nullptr, {});
  auto is_seq_var2 = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var2 != nullptr, {});
  VectorRef tuple_ref2({is_tuple_getitem2, split_ref, is_seq_var2});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  VectorRef sigmoid_ref({is_activation, tuple_ref2});
  auto is_mul = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>);
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  VectorRef mul_ref({is_mul, tuple_ref1, sigmoid_ref});
  return mul_ref;
}

const AnfNodePtr GLUFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  MS_LOG(DEBUG) << "glu_fusion pass";
  if (!utils::isa<CNodePtr>(node)) {
    return nullptr;
  }
  if (IsMarkedTrainOp(utils::cast<CNodePtr>(node))) {
    return nullptr;
  }
  auto cnode = CreateGLUNode(func_graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "new glu node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
