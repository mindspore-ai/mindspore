/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "pre_activate/ascend/format_type/insert_cast_for_runop.h"

#include <memory>

#include "device/kernel_info.h"
#include "pre_activate/ascend/ascend_helper.h"
#include "pre_activate/common/helper.h"
#include "kernel/oplib/oplib.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
const BaseRef RunOpInsertCast::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr RunOpInsertCast::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfAlgo::IsRealCNodeKernel(node) || func_graph == nullptr) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  // process input
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return InsertCastForInput(func_graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
