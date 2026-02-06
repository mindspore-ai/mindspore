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
#include "backend/common/pass/other/process_call_inline.h"
#include <memory>
#include <string>
#include <vector>
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/utils/anfalgo.h"
#include "primitive/framework_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckCallInline(const CNodePtr &cnode) {
  if (!common::AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimCall)) {
    return false;
  }
  auto call_graph = cnode->input(kIndex1);
  auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
  MS_EXCEPTION_IF_NULL(sub_kernel_graph);
  return sub_kernel_graph->need_inline();
}
}  // namespace

std::vector<std::string> ProcessCallInline::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimCall->name()};
  return ret;
}

const BaseRef ProcessCallInline::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimCall, Xs});
}

const AnfNodePtr ProcessCallInline::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (CheckCallInline(cnode)) {
    auto call_graph = cnode->input(kIndex1);
    auto sub_kernel_graph = session::AnfRuntimeAlgorithm::GetValueNodeKernelGraph(call_graph);
    std::vector<AnfNodePtr> call_inputs = {};
    call_inputs.push_back(NewValueNode(std::make_shared<Primitive>(prim::kPrimCallInline->name())));
    for (size_t i = kIndex1; i < common::AnfAlgo::GetInputNum(cnode); i++) {
      (void)call_inputs.emplace_back(common::AnfAlgo::GetInputNode(cnode, i));
    }
    auto new_call_node = NewCNode(call_inputs, graph);
    MS_EXCEPTION_IF_NULL(new_call_node);
    new_call_node->set_scope(node->scope());
    new_call_node->set_abstract(cnode->abstract());
    common::AnfAlgo::SetNodeAttr(kAttrKernelGraph, MakeValue(sub_kernel_graph), new_call_node);
    new_call_node->set_attrs(cnode->attrs());
    return new_call_node;
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
