/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/graph_kernel_expander_lite.h"

#include <utility>
#include <algorithm>
#include <vector>
#include <map>
#include <string>

#include "common/graph_kernel/core/graph_kernel_callback.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
AnfNodePtr ParaToValueDeco::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  for (const auto &idx : input_idx_) {
    if (cnode->input(idx + 1)->isa<Parameter>()) {
      auto param_value = cnode->input(idx + 1)->cast<ParameterPtr>()->default_param()->cast<tensor::TensorPtr>();
      auto int_value = static_cast<int *>(param_value->data_ptr()->data());
      ShapeVector out_list;
      std::transform(int_value, int_value + param_value->data_ptr()->size(), std::back_inserter(out_list), IntToLong);
      auto value = std::make_shared<ValueNode>(MakeValue(out_list));
      cnode->set_input(idx + 1, value);
    }
  }
  return decorated_->Run(cnode);
}

AnfNodePtr FixFormatDeco::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  std::vector<std::string> format = {kOpFormat_DEFAULT};
  auto ori_format = node->cast<CNodePtr>()->GetAttr(kOutputsFormat);
  cnode->AddAttr(kOutputsFormat, MakeValue(format));
  auto ret = decorated_->Run(cnode);
  if (ret == nullptr) {
    cnode->AddAttr(kOutputsFormat, ori_format);
    return nullptr;
  }
  auto fg = GetCNodeFuncGraph(ret);
  for (auto sub_cnode : fg->GetOrderedCnodes()) {
    sub_cnode->AddAttr(kOutputsFormat, ori_format);
  }
  auto ret_cnode = ret->cast<CNodePtr>();
  ret_cnode->AddAttr(kOutputsFormat, ori_format);
  return ret_cnode;
}

std::vector<PrimitivePtr> GraphKernelExpanderLite::InitOpList() {
  std::vector<OpWithLevel> expand_ops_with_level = {
    {kCPUDevice, OpLevel_0, prim::kPrimAddFusion},    {kCPUDevice, OpLevel_0, prim::kPrimMulFusion},
    {kCPUDevice, OpLevel_0, prim::kPrimSubFusion},    {kCPUDevice, OpLevel_0, prim::kPrimSquare},
    {kCPUDevice, OpLevel_1, prim::kPrimReduceFusion}, {kCPUDevice, OpLevel_0, prim::kPrimActivation},
    {kCPUDevice, OpLevel_0, prim::kPrimDivFusion},    {kCPUDevice, OpLevel_1, prim::kPrimExpandDims},
    {kCPUDevice, OpLevel_0, prim::kPrimExpFusion},    {kCPUDevice, OpLevel_1, prim::kPrimSqueeze},
    {kCPUDevice, OpLevel_1, prim::kPrimTranspose},    {kCPUDevice, OpLevel_1, prim::kPrimReshape},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(expand_ops_with_level, flags.fusion_ops_level, flags.enable_expand_ops_only,
                              flags.enable_expand_ops, flags.disable_expand_ops);
}

bool GraphKernelExpanderLite::CanExpand(const CNodePtr &node) const {
  if (!GraphKernelExpander::CanExpand(node)) {
    return false;
  }
  // check if the node has dynamic shape
  auto cb = Callback::Instance();
  for (size_t i = 0; i < node->size() - 1; i++) {
    if (!node->input(i + 1)->isa<Parameter>() && !node->input(i + 1)->isa<ValueNode>() &&
        cb->GetInputShape(node, i).size() == 0) {
      MS_LOG(INFO) << "cnode with no input info can not expand now, node is " << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

ExpanderPtr GraphKernelExpanderLite::InitExpander(const AnfNodePtr &node) {
  auto expander = std::make_shared<DefaultExpander>(Callback::Instance());
  std::map<std::string, ExpanderCreatorFuncList> creators = {
    {prim::kPrimReduceFusion->name(), {InputToAttrDeco::GetCreator({1}), FixFormatDeco::Creator}},
    {prim::kPrimExpandDims->name(), {InputToAttrDeco::GetCreator({1}), FixFormatDeco::Creator}},
    {prim::kPrimSqueeze->name(), {FixFormatDeco::Creator}},
    {prim::kPrimReshape->name(), {InputToAttrDeco::GetCreator({1}), FixFormatDeco::Creator}},
    {prim::kPrimTranspose->name(), {ParaToValueDeco::GetCreator({1}), InputToAttrDeco::GetCreator({1})}},
  };
  auto iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    return WrapExpander(expander, iter->second);
  }
  return expander;
}
}  // namespace mindspore::graphkernel
