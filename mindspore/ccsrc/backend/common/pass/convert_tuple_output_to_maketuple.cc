/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/convert_tuple_output_to_maketuple.h"

#include <algorithm>
#include <memory>

#include "backend/common/optimizer/helper.h"
#include "backend/common/session/kernel_graph.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr ConvertTupleInputToMakeTuple(const FuncGraphPtr &graph, const AnfNodePtr &tuple_anf) {
  MS_EXCEPTION_IF_NULL(tuple_anf);
  MS_EXCEPTION_IF_NULL(graph);

  if (!common::AnfAlgo::IsTupleOutput(tuple_anf)) {
    return tuple_anf;
  }
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  FuncGraphPtr anf_graph = tuple_anf->func_graph();
  if (anf_graph != nullptr) {
    kernel_graph = anf_graph->cast<KernelGraphPtr>();
  }
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (kernel_graph->FindTupleParameterToMakeTupleMap(tuple_anf)) {
    return kernel_graph->FindTupleParameterToMakeTupleMap(tuple_anf);
  }
  auto make_tuple = kernel_graph->TransTupleToMakeTuple(tuple_anf);
  MS_EXCEPTION_IF_NULL(make_tuple);
  kernel_graph->InsertTupleParameterToMakeTupleMap(tuple_anf, make_tuple);
  // replace graph inputs if input is a parameter
  kernel_graph->ReplaceGraphInput(tuple_anf, make_tuple);
  return make_tuple;
}

bool IsKerenlGraphOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  const auto &outputs =
    common::AnfAlgo::GetAllOutputIndexByReturnTypes(func_graph->output(), {prim::kPrimTupleGetItem});
  return std::find_if(outputs.begin(), outputs.end(), [&node](const auto &output) { return output.first == node; }) !=
         outputs.end();
}

bool IsNeedConvert(const FuncGraphPtr &func_graph, const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  return (input->Type() != nullptr && AnfUtils::IsRealKernel(input) && common::AnfAlgo::IsTupleOutput(input) &&
          !common::AnfAlgo::CheckPrimitiveType(input, prim::kPrimCall) &&
          (input->isa<Parameter>() || input->isa<ValueNode>() || IsKerenlGraphOutput(func_graph, input)) &&
          (!common::AnfAlgo::IsDynamicSequence(input)));
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
  if (IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
    auto real_input = common::AnfAlgo::GetTupleGetItemRealInput(cnode);
    MS_EXCEPTION_IF_NULL(real_input);
    if (!real_input->isa<Parameter>() && !real_input->isa<ValueNode>()) {
      return nullptr;
    }
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimUpdateState)) {
    return nullptr;
  }

  bool cnode_input_changed = false;
  for (size_t i = 0; i < cnode->inputs().size(); ++i) {
    const auto &input = cnode->inputs()[i];
    if (IsNeedConvert(func_graph, input)) {
      MS_LOG(INFO) << "Convert tuple input to make tuple for node:" << node->fullname_with_scope()
                   << ", input node:" << input->fullname_with_scope();
      auto new_input = ConvertTupleInputToMakeTuple(func_graph, input);
      if (new_input->isa<CNode>() && common::AnfAlgo::CheckPrimitiveType(new_input, prim::kPrimMakeTuple)) {
        auto make_tuple = new_input->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(make_tuple);
        if (make_tuple->inputs().size() == 1) {
          new_input = input;
        }
      }
      cnode->set_input(i, new_input);
      cnode_input_changed = true;
    }
  }
  FuncGraphPtr graph = node->func_graph();
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr || !cnode_input_changed) {
    return nullptr;
  }
  return NewCNode(cnode, kernel_graph);
}
}  // namespace opt
}  // namespace mindspore
