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
#include "plugin/device/ascend/optimizer/ge/process_call_inline.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckCallInline(const CNodePtr &cnode) {
  if (!common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    return false;
  }
  auto call_graph = cnode->input(kIndex1);
  auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
  return sub_kernel_graph->need_inline();
}
}  // namespace

const BaseRef ProcessCallInline::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimCall, Xs});
}

const AnfNodePtr ProcessCallInline::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckCallInline(cnode)) {
    auto call_graph = cnode->input(kIndex1);
    auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
    std::vector<AnfNodePtr> call_inline_inputs = {
      NewValueNode(std::make_shared<Primitive>(prim::kPrimCallInline->name()))};
    for (size_t i = kIndex1; i < common::AnfAlgo::GetInputNum(cnode); i++) {
      call_inline_inputs.emplace_back(common::AnfAlgo::GetInputNode(cnode, i));
    }
    auto call_inline = graph->NewCNode(call_inline_inputs);
    MS_EXCEPTION_IF_NULL(call_inline);
    call_inline->set_abstract(cnode->abstract());
    common::AnfAlgo::SetNodeAttr(kAttrKernelGraph, MakeValue(sub_kernel_graph), call_inline);
    return call_inline;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
