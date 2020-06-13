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

#include "pre_activate/pass/erase_visit_attr.h"
#include <memory>
#include <vector>
#include "kernel/common_utils.h"
#include "session/anf_runtime_algorithm.h"
#include "pre_activate/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef EraseVisitAttr::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(Visited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr EraseVisitAttr::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  if (node != nullptr && AnfAlgo::IsRealCNodeKernel(node)) {
    if (AnfAlgo::IsCompositeKernel(node)) {
      auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
      MS_EXCEPTION_IF_NULL(fg);
      std::vector<AnfNodePtr> todos;
      kernel::GetValidKernelNodes(fg, &todos);
      for (auto &t : todos) {
        AnfAlgo::EraseNodeAttr(kAttrVisited, t);
      }
    }
    AnfAlgo::EraseNodeAttr(kAttrVisited, node);
  } else {
    AnfAlgo::EraseNodeAttr(kAttrVisited, node);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
