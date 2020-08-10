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
#include "backend/optimizer/pass/convert_tuple_output_to_maketuple.h"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/helper.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr ConvertTupleInputToMakeTuple(const FuncGraphPtr &graph, const AnfNodePtr &tuple_anf,
                                        std::unordered_map<AnfNodePtr, AnfNodePtr> *transed_nodes) {
  MS_EXCEPTION_IF_NULL(tuple_anf);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(transed_nodes);

  if (!AnfAlgo::IsTupleOutput(tuple_anf)) {
    return tuple_anf;
  }
  auto transed_node_it = transed_nodes->find(tuple_anf);
  if (transed_node_it != transed_nodes->end()) {
    return transed_node_it->second;
  }
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  auto make_tuple = kernel_graph->TransTupleToMakeTuple(tuple_anf);
  (*transed_nodes)[tuple_anf] = make_tuple;
  // replace graph inputs if input is a parameter
  kernel_graph->ReplaceGraphInput(tuple_anf, make_tuple);
  return make_tuple;
}
}  // namespace

const BaseRef ConvertTupleOutputToMaketuple::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr ConvertTupleOutputToMaketuple::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                        const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::unordered_map<AnfNodePtr, AnfNodePtr> transed_nodes;
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    auto real_input = AnfAlgo::GetTupleGetItemRealInput(cnode);
    MS_EXCEPTION_IF_NULL(real_input);
    if (!real_input->isa<Parameter>() && !real_input->isa<ValueNode>()) {
      return nullptr;
    }
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimControlDepend)) {
    return nullptr;
  }
  bool cnode_input_changed = false;
  for (size_t i = 0; i < cnode->inputs().size(); ++i) {
    const auto &input = cnode->inputs()[i];
    if (input->Type() != nullptr && AnfAlgo::IsRealKernel(input) && AnfAlgo::IsTupleOutput(input) &&
        !AnfAlgo::CheckPrimitiveType(input, prim::kPrimCall)) {
      cnode->set_input(i, ConvertTupleInputToMakeTuple(func_graph, input, &transed_nodes));
      cnode_input_changed = true;
    }
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr || !cnode_input_changed) {
    return nullptr;
  }
  return kernel_graph->NewCNode(cnode);
}
}  // namespace opt
}  // namespace mindspore
