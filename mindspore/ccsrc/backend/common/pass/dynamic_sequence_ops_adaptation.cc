/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/dynamic_sequence_ops_adaptation.h"
#include <set>
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
const AnfNodePtr DynamicSequenceOpsAdaptation::Process(const FuncGraphPtr &, const AnfNodePtr &node,
                                                       const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return node;
  }
  static const std::set<std::string> kDynamicTupleInputOpNames = {
    kSplitOpName, kConcatOpName, kAddNOpName, kStackOpName, kPackOpName, kUnstackOpName, kUnpackOpName};
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto iter = kDynamicTupleInputOpNames.find(op_name);
  if (iter == kDynamicTupleInputOpNames.end()) {
    return node;
  }

  if (common::AnfAlgo::HasDynamicTupleInput(cnode) || common::AnfAlgo::IsDynamicSequence(cnode)) {
    auto primitive = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(primitive);
    constexpr auto kDynamicTupleInputOpNamePrefix = "Sequence";
    auto new_op_name = kDynamicTupleInputOpNamePrefix + op_name;
    primitive->set_name(new_op_name);
    // reset full scope name
    cnode->set_fullname_with_scope("");
    MS_LOG(INFO) << "Rename op type from " << op_name << " to " << new_op_name << " for op "
                 << cnode->fullname_with_scope();
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
  }

  return node;
}
}  // namespace opt
}  // namespace mindspore
