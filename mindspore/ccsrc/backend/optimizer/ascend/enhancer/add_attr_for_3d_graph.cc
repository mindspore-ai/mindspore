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

#include "backend/optimizer/ascend/enhancer/add_attr_for_3d_graph.h"
#include <memory>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "base/core_ops.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
const BaseRef AddIoFormatAttrFor3DGraph::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr AddIoFormatAttrFor3DGraph::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  if (AnfAlgo::IsRealKernel(node)) {
    AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
    auto formats = AnfAlgo::GetAllOutputFormats(node);
    if (std::any_of(formats.begin(), formats.end(),
                    [](const std::string &format) { return k3DFormatSet.find(format) != k3DFormatSet.end(); })) {
      AnfAlgo::SetNodeAttr(kAttrFormat, MakeValue(kOpFormat_NCDHW), node);
    }
    return node;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
