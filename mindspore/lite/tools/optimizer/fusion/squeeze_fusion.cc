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

#include "tools/optimizer/fusion/squeeze_fusion.h"
#include <memory>
#include "schema/inner/model_generated.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"

namespace mindspore::opt {
const BaseRef SqueezeFusion::DefinePattern() const {
  auto is_squeeze = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSqueeze>);
  MS_CHECK_TRUE_RET(is_squeeze != nullptr, {});
  auto is_bn = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimFusedBatchNorm>);
  MS_CHECK_TRUE_RET(is_bn != nullptr, {});
  auto is_param1 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param1 != nullptr, {});
  auto is_param2 = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param2 != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  VectorRef bn_ref = VectorRef({is_bn, is_squeeze, is_param1, is_param2, is_seq_var});
  auto is_activation = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimActivation>);
  MS_CHECK_TRUE_RET(is_activation != nullptr, {});
  VectorRef act_ref = VectorRef({is_activation, bn_ref});
  auto is_unsqueeze = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimUnsqueeze>);
  MS_CHECK_TRUE_RET(is_unsqueeze != nullptr, {});
  return VectorRef({is_unsqueeze, act_ref});
}

const AnfNodePtr SqueezeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &unsqueeze_node,
                                        const EquivPtr &) const {
  if (func_graph == nullptr || unsqueeze_node == nullptr) {
    return nullptr;
  }

  auto unsqueeze_cnode = unsqueeze_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(unsqueeze_cnode != nullptr, nullptr);
  auto act_node = unsqueeze_cnode->input(1);
  if (act_node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  auto bn_node = act_node->cast<CNodePtr>()->input(1);
  if (bn_node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  auto squeeze_node = bn_node->cast<CNodePtr>()->input(1);
  if (squeeze_node->cast<CNodePtr>() == nullptr) {
    return nullptr;
  }
  auto pre_node = squeeze_node->cast<CNodePtr>()->input(1);

  if (GetCNodePrimitive(unsqueeze_node)->GetAttr(ops::kAxis) ==
      GetCNodePrimitive(unsqueeze_node)->GetAttr(ops::kAxis)) {
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    (void)manager->Replace(unsqueeze_node, act_node);
    (void)manager->Replace(squeeze_node, pre_node);
    return pre_node;
  }
  return nullptr;
}
}  // namespace mindspore::opt
