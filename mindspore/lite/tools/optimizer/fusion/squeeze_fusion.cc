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

namespace mindspore::opt {
const BaseRef SqueezeFusion::DefinePattern() const {
  auto squeeze_var = std::make_shared<CondVar>(IsSqueezeNode);
  auto act_var = std::make_shared<CondVar>(IsActivationNode);
  VectorRef act_ref = VectorRef({act_var, squeeze_var});
  auto unsqueeze_var = std::make_shared<CondVar>(IsSqueezeNode);
  return VectorRef({unsqueeze_var, act_ref});
}

const AnfNodePtr SqueezeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &unsqueeze_node,
                                        const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(unsqueeze_node) != lite::RET_OK) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return nullptr;
  }

  auto act_node = unsqueeze_node->cast<CNodePtr>()->input(1);
  if (CheckIfCNodeIsNull(act_node->cast<CNodePtr>()) != lite::RET_OK) {
    return nullptr;
  }
  auto squeeze_node = act_node->cast<CNodePtr>()->input(1);
  if (CheckIfCNodeIsNull(squeeze_node->cast<CNodePtr>()) != lite::RET_OK) {
    return nullptr;
  }
  auto pre_node = squeeze_node->cast<CNodePtr>()->input(1);
  if (CheckIfCNodeIsNull(pre_node->cast<CNodePtr>()) != lite::RET_OK) {
    return nullptr;
  }

  if (GetCNodePrimitive(unsqueeze_node)->GetAttr("axis") == GetCNodePrimitive(unsqueeze_node)->GetAttr("axis")) {
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    manager->Replace(unsqueeze_node, act_node);
    manager->Replace(squeeze_node, pre_node);
    return pre_node;
  }
  return nullptr;
}
}  // namespace mindspore::opt
