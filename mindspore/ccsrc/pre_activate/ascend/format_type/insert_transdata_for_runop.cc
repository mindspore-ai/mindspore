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

#include "pre_activate/ascend/format_type/insert_transdata_for_runop.h"
#include <memory>
#include "utils/utils.h"
#include "pre_activate/ascend/ascend_helper.h"
#include "session/anf_runtime_algorithm.h"
#include "device/kernel_info.h"
#include "kernel/oplib/oplib.h"

namespace mindspore {
namespace opt {
const BaseRef RunOpInsertTransData::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  MS_EXCEPTION_IF_NULL(V);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  MS_EXCEPTION_IF_NULL(Xs);
  return VectorRef({V, Xs});
}

const AnfNodePtr RunOpInsertTransData::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  if (node == nullptr || !AnfAlgo::IsRealKernel(node)) {
    return nullptr;
  }
  AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  MS_LOG(DEBUG) << "====process op: " << node->DebugString();
  return InsertTransOpForInput(func_graph, node, kernel_select_);
}
}  // namespace opt
}  // namespace mindspore
