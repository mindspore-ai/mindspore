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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/activation_fusion.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/fusion/activation.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace opt {
STATUS DoFusion(CNodePtr cur_cnode, const CNodePtr &pre_cnode) {
  auto cur_act_prim = ops::GetOperator<mindspore::ops::Activation>(cur_cnode->input(0));
  MS_ASSERT(cur_act_prim != nullptr);
  auto pre_act_prim = ops::GetOperator<mindspore::ops::Activation>(pre_cnode->input(0));
  MS_ASSERT(pre_act_prim != nullptr);
  MS_CHECK_TRUE_MSG(cur_act_prim->GetAttr(ops::kActivationType) != nullptr, RET_ERROR, "Get activation type failed.");
  MS_CHECK_TRUE_MSG(pre_act_prim->GetAttr(ops::kActivationType) != nullptr, RET_ERROR, "Get activation type failed.");
  auto cur_act_type = cur_act_prim->get_activation_type();
  auto pre_act_type = pre_act_prim->get_activation_type();
  if (!(cur_act_type == HARD_TANH || cur_act_type == RELU || cur_act_type == RELU6)) {
    return lite::RET_NOT_SUPPORT;
  }
  if (!(pre_act_type == HARD_TANH || pre_act_type == RELU || pre_act_type == RELU6)) {
    return lite::RET_NOT_SUPPORT;
  }
  MS_CHECK_TRUE_MSG(pre_act_prim->GetAttr(ops::kMaxVal) != nullptr, RET_ERROR, "Get max value failed.");
  MS_CHECK_TRUE_MSG(cur_act_prim->GetAttr(ops::kMaxVal) != nullptr, RET_ERROR, "Get max value failed.");
  MS_CHECK_TRUE_MSG(pre_act_prim->GetAttr(ops::kMinVal) != nullptr, RET_ERROR, "Get min value failed.");
  MS_CHECK_TRUE_MSG(cur_act_prim->GetAttr(ops::kMinVal) != nullptr, RET_ERROR, "Get min value failed.");
  auto pre_max_val =
    pre_act_type == RELU ? FLT_MAX : pre_act_type == RELU6 ? kValueThreshold6 : pre_act_prim->get_max_val();
  auto pre_min_val = (pre_act_type == RELU || pre_act_type == RELU6) ? 0 : pre_act_prim->get_min_val();
  auto cur_max_val =
    cur_act_type == RELU ? FLT_MAX : cur_act_type == RELU6 ? kValueThreshold6 : cur_act_prim->get_max_val();
  auto cur_min_val = (cur_act_type == RELU || cur_act_type == RELU6) ? 0 : cur_act_prim->get_min_val();
  auto new_max_val = std::min(pre_max_val, cur_max_val);
  auto new_min_val = std::max(pre_min_val, cur_min_val);
  MS_CHECK_TRUE_MSG(new_min_val <= new_max_val, RET_ERROR,
                    "The min value is larger than the max value, fusion failed.");
  cur_act_prim->set_min_val(new_min_val);
  cur_act_prim->set_max_val(new_max_val);
  cur_act_prim->set_activation_type(HARD_TANH);
  if (new_min_val == 0 && new_max_val == kValueThreshold6) {
    cur_act_prim->set_activation_type(RELU6);
  }
  if (new_min_val == 0 && new_max_val == FLT_MAX) {
    cur_act_prim->set_activation_type(RELU);
  }
  return lite::RET_OK;
}

bool ActivationFusion::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNode>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!CheckPrimitiveType(node, prim::kPrimActivation)) {
      continue;
    }
    MS_CHECK_TRUE_RET(cnode->size() == kInputSizeTwo, false);
    if (!CheckPrimitiveType(cnode->input(SECOND_INPUT), prim::kPrimActivation)) {
      continue;
    }
    if (IsMarkedTrainOp(cnode)) {
      return false;
    }
    auto pre_act_cnode = cnode->input(SECOND_INPUT)->cast<CNodePtr>();
    MS_CHECK_TRUE_RET(pre_act_cnode != nullptr, false);
    if (IsMultiOutputTensors(func_graph, pre_act_cnode)) {
      MS_LOG(WARNING) << "activation node is used as input by multiple cnodes, Fusion failed! ,name:"
                      << pre_act_cnode->fullname_with_scope();
      return false;
    }
    auto ret = DoFusion(cnode, pre_act_cnode);
    if (ret != RET_OK) {
      return false;
    }
    auto manager = func_graph->manager();
    MS_ASSERT(manager != nullptr);
    MS_CHECK_TRUE_RET(pre_act_cnode->size() == kInputSizeTwo, false);
    auto pre_node = pre_act_cnode->input(SECOND_INPUT);
    (void)manager->Replace(pre_act_cnode, pre_node);
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
