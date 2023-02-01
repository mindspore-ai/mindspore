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
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "backend/common/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
const BaseRef AscendConvertTupleInputToDynamicInput::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr AscendConvertTupleInputToDynamicInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                                const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // this pass should be in front of concat_fission, pack_fission, addn_fission, since the input should be unfold before
  // this passes.
  // the auto_monad pass should before this pass
  bool is_communication_op = common::AnfAlgo::IsCommunicationOp(node);
  static const PrimitiveSet need_unfold_node = {prim::kPrimAddN,        prim::kPrimConcatD,    prim::kPrimPack,
                                                prim::kPrimStack,       prim::kPrimCallInline, prim::kPrimPrint,
                                                prim::kPrimSwitchLayer, prim::kPrimCall,       prim::kPrimSwitch};
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  if (!is_communication_op && need_unfold_node.find(prim) == need_unfold_node.end()) {
    return nullptr;
  }

  return ConvertMakeTupleInputToPlantInputs(func_graph, node->cast<CNodePtr>());
}
}  // namespace opt
}  // namespace mindspore
