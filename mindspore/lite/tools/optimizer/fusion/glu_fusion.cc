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
#include "ops/fusion/activation.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore {
namespace opt {
CNodePtr GLUFusion::CreateGLUNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto glu_prim = std::make_shared<ops::Activation>();
  glu_prim->set_activation_type(mindspore::GLU);
  auto input_node = utils::cast<AnfNodePtr>((*equiv)[input_]);
  MS_ASSERT(input_node != nullptr);
  auto glu_cnode = func_graph->NewCNode(glu_prim, {input_node});
  glu_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_glu");
  glu_cnode->set_abstract(node->abstract()->Clone());
  return glu_cnode;
}

const BaseRef GLUFusion::DefinePattern() const {
  VectorRef split_ref(
    {std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSplit>), input_, std::make_shared<SeqVar>()});
  VectorRef tuple_ref1(
    {std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>), split_ref, std::make_shared<SeqVar>()});
  VectorRef tuple_ref2(
    {std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimTupleGetItem>), split_ref, std::make_shared<SeqVar>()});
  VectorRef sigmoid_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>), tuple_ref2});
  VectorRef mul_ref({std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimMulFusion>), tuple_ref1, sigmoid_ref});
  return mul_ref;
}

const AnfNodePtr GLUFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                    const EquivPtr &equiv) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  MS_ASSERT(equiv != nullptr);
  MS_LOG(DEBUG) << "glu_fusion pass";
  if (!utils::isa<CNodePtr>(node)) {
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
