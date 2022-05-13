/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/expander.h"

#include <string>
#include <vector>
#include <map>
#include "mindspore/core/utils/anf_utils.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "pybind_api/ir/primitive_py.h"
#include "common/graph_kernel/adapter/expander.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
void ConvertPrimToPrimPy(const FuncGraphPtr &graph) {
  auto todos = TopoSort(graph->get_return());
  for (const auto &node : todos) {
    if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto primitive = GetCNodePrimitive(node);
    if (!primitive || dyn_cast<PrimitivePy>(primitive)) {
      continue;
    }
    parallel::OperatorAttrs attrs;
    auto new_prim = parallel::CreateOpInstance(attrs, primitive->name(), "")->cast<PrimitivePtr>();
    (void)new_prim->SetAttrs(primitive->attrs());
    AnfNodePtrList inputs = {NewValueNode(new_prim)};
    auto cnode = dyn_cast<CNode>(node);
    inputs.insert(inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
    cnode->set_inputs(inputs);
  }
}

FuncGraphPtr TryExpandCNodeFE(const AnfNodePtr &node) {
#ifdef ENABLE_AKG
  using graphkernel::InputToAttrDeco;
  auto primitive = GetCNodePrimitive(node);
  if (primitive == nullptr) return nullptr;
  auto expander = graphkernel::GetExpander(node);
  std::map<std::string, graphkernel::ExpanderCreatorFuncList> creators = {
    {prim::kPrimExpandDims->name(), {InputToAttrDeco::GetCreator({1})}},
    {prim::kPrimReshape->name(), {InputToAttrDeco::GetCreator({1})}},
    {prim::kPrimReduceMean->name(), {InputToAttrDeco::GetCreator({1})}},
    {prim::kPrimGather->name(), {InputToAttrDeco::GetCreator({2})}},
    {kTileOpName, {InputToAttrDeco::GetCreator({1})}},
    {kSliceOpName, {InputToAttrDeco::GetCreator({1, 2})}},
  };
  auto iter = creators.find(GetCNodePrimitive(node)->name());
  if (iter != creators.end()) {
    expander = graphkernel::WrapExpander(expander, iter->second);
  }
  expander = graphkernel::AttrToInputDeco::Creator(expander);
  auto fg = GetCNodeFuncGraph(expander->Run(node));
  if (fg == nullptr) return nullptr;
  ConvertPrimToPrimPy(fg);
  return fg;
#else
  return nullptr;
#endif
}
}  // namespace opt
}  // namespace mindspore
