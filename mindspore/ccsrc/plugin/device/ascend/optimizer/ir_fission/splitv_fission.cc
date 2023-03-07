/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/splitv_fission.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore::opt {
const BaseRef SplitVFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto split_prim = std::make_shared<Primitive>(kSplitVDOpName);
  return VectorRef({split_prim, Xs});
}

const AnfNodePtr SplitVFission::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (common::AnfAlgo::IsDynamicShape(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Check output num
  if (!common::AnfAlgo::HasNodeAttr(kAttrNumSplit, cnode)) {
    return nullptr;
  }
  auto num_split = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrNumSplit);
  if (num_split <= outputs_divisor_) {
    return nullptr;
  }
  return DoFission(func_graph, cnode, num_split, outputs_divisor_,
                   common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrSplitDim));
}
}  // namespace mindspore::opt
