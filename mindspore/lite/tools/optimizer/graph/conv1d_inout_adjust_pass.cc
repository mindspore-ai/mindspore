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
#include "tools/optimizer/graph/conv1d_inout_adjust_pass.h"
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include "mindspore/lite/include/errorcode.h"
#include "ops/conv2d.h"
#include "ops/squeeze.h"
#include "ops/unsqueeze.h"
#include "ops/primitive_c.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::opt {
CNodePtr Conv1DInOutAdjustPass::NewUnsqueezeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node,
                                                   const std::vector<int64_t> &axis) {
  auto unsqueeze_prim = std::make_shared<ops::Unsqueeze>();
  if (unsqueeze_prim == nullptr) {
    MS_LOG(ERROR) << "create unsqueeze failed.";
    return nullptr;
  }
  unsqueeze_prim->set_attr("axis", MakeValue(axis));
  ValueNodePtr value_node = NewValueNode(unsqueeze_prim);
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto unsqueeze = func_graph->NewCNode(op_inputs);
  unsqueeze->set_fullname_with_scope(input_node->fullname_with_scope() + "_unsqueeze");
  return unsqueeze;
}

CNodePtr Conv1DInOutAdjustPass::NewSqueezeOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr input_node,
                                                 const std::vector<int64_t> &axis) {
  auto squeeze_prim = std::make_shared<ops::Squeeze>();
  if (squeeze_prim == nullptr) {
    MS_LOG(ERROR) << "create squeeze failed.";
    return nullptr;
  }
  squeeze_prim->set_attr("axis", MakeValue(axis));
  ValueNodePtr value_node = NewValueNode(squeeze_prim);
  std::vector<AnfNodePtr> op_inputs = {value_node, input_node};
  auto squeeze = func_graph->NewCNode(op_inputs);
  squeeze->set_fullname_with_scope(input_node->fullname_with_scope() + "_squeeze");
  return squeeze;
}

bool Conv1DInOutAdjustPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto cnodes = func_graph->GetOrderedCnodes();
  for (auto &cnode : cnodes) {
    if (!CheckPrimitiveType(cnode, prim::kPrimConv2D) && !CheckPrimitiveType(cnode, prim::kPrimConv2DFusion)) {
      continue;
    }
    auto conv2d_node = GetValueNode<std::shared_ptr<mindspore::ops::Conv2D>>(cnode->input(0));
    if (conv2d_node == nullptr) {
      MS_LOG(ERROR) << "conv2d is nullptr.";
      return false;
    }
    if (conv2d_node->GetAttr(ops::kFormat) == nullptr) {
      MS_LOG(ERROR) << "The format of conv2d is nullptr.";
      return false;
    }

    std::vector<int64_t> axis;
    switch (conv2d_node->get_format()) {
      case mindspore::Format::NWC:
        axis = {1};
        break;
      case mindspore::Format::NCW:
        axis = {2};
        break;
      default:
        continue;
    }

    auto input_node = cnode->input(1);
    auto unsqueeze = NewUnsqueezeOpNode(func_graph, input_node, axis);
    if (unsqueeze == nullptr) {
      MS_LOG(ERROR) << "New unsqueeze node failed.";
      return false;
    }
    manager->Replace(input_node, unsqueeze);
    auto squeeze = NewSqueezeOpNode(func_graph, cnode, axis);
    if (squeeze == nullptr) {
      MS_LOG(ERROR) << "New squeeze node failed.";
      return false;
    }
    manager->Replace(cnode, squeeze);
  }
  return true;
}
}  // namespace mindspore::opt
