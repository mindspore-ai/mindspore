/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ge/process_partial_inline.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/anf_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
const BaseRef ProcessPartialInline::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPartial, Xs});
}

const AnfNodePtr ProcessPartialInline::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                               const EquivPtr &) const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (!context->IsKByKExecutorMode()) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->HasPrimalAttr(kAttrNotCut)) {
    return nullptr;
  }
  auto partial_graph = cnode->input(kIndex1);
  auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(partial_graph);
  std::vector<AnfNodePtr> partial_inline_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimPartialInline->name()))};
  for (size_t i = kIndex1; i < common::AnfAlgo::GetInputNum(cnode); i++) {
    partial_inline_inputs.emplace_back(common::AnfAlgo::GetInputNode(cnode, i));
  }
  auto partial_inline = graph->NewCNode(partial_inline_inputs);
  MS_EXCEPTION_IF_NULL(partial_inline);
  partial_inline->set_abstract(cnode->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrKernelGraph, MakeValue(sub_kernel_graph), partial_inline);
  return partial_inline;
}
}  // namespace opt
}  // namespace mindspore