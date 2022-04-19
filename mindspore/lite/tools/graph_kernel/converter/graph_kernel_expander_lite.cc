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
#include <vector>
#include <map>
#include <string>

#include "backend/common/optimizer/const_input_to_attr.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"

namespace mindspore::graphkernel {
AnfNodePtr InputToAttrDeco::Run(const AnfNodePtr &node) {
  auto cnode = QuickCloneCNode(node);
  opt::ConstInputToAttr(cnode, input_idx_);
  return decorated_->Run(cnode);
}

std::vector<PrimitivePtr> GraphKernelExpanderLite::InitOpList() {
  std::vector<OpWithLevel> expand_ops_with_level = {
    {kCPUDevice, OpLevel_0, prim::kPrimAddFusion},    {kCPUDevice, OpLevel_0, prim::kPrimMulFusion},
    {kCPUDevice, OpLevel_0, prim::kPrimSubFusion},    {kCPUDevice, OpLevel_0, prim::kPrimSquare},
    {kCPUDevice, OpLevel_1, prim::kPrimReduceFusion}, {kCPUDevice, OpLevel_0, prim::kPrimActivation},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(expand_ops_with_level, flags.fusion_ops_level, flags.enable_expand_ops_only,
                              flags.enable_expand_ops, flags.disable_expand_ops);
}

ExpanderPtr GraphKernelExpanderLite::InitExpander(const AnfNodePtr &node) {
  auto expander = std::make_shared<DefaultExpander>(Callback::Instance());
  std::map<std::string, ExpanderCreatorFuncList> creators = {
    {prim::kPrimReduceFusion->name(), {InputToAttrDeco::GetCreator({1})}},
  };
  auto iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    return WrapExpander(expander, iter->second);
  }
  return expander;
}
}  // namespace mindspore::graphkernel
