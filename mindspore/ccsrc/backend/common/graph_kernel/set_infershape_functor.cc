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
#include "backend/common/graph_kernel/set_infershape_functor.h"

#include <memory>
#include <vector>

#include "backend/common/graph_kernel/symbol_engine/symbol_engine.h"
#include "include/common/utils/anfalgo.h"
#include "ir/anf.h"

namespace mindspore::graphkernel {
BaseShapePtr SymbolEngineInfer::InferShape(const CNodePtr &cnode, const AbstractBasePtrList &args) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Infer shape using symbol engine for cnode: " << cnode->fullname_with_scope();
  auto func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto output = func_graph->output();
  auto symbol_engine = GetValue<SymbolEnginePtr>(func_graph->get_attr(kAttrSymbolEngine));
  MS_EXCEPTION_IF_NULL(symbol_engine);
  if (!symbol_engine->Infer(args)) {
    MS_LOG(WARNING) << "Infer failed by symbol engine. node " << cnode->fullname_with_scope();
    return nullptr;
  }
  auto out_shapes = symbol_engine->QueryShape(output);
  if (cnode->abstract()->isa<abstract::AbstractTuple>()) {
    abstract::BaseShapePtrList shapes(out_shapes.size());
    (void)std::transform(out_shapes.begin(), out_shapes.end(), shapes.begin(),
                         [](const ShapeVector &s) { return std::make_shared<abstract::TensorShape>(s); });
    return std::make_shared<abstract::TupleShape>(shapes);
  }
  return std::make_shared<abstract::TensorShape>(out_shapes.front());
}

bool SetInferShapeFunctor::Run(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto cnodes = TopoSort(func_graph->output(), SuccIncoming,
                         [](const AnfNodePtr &node) { return node->isa<CNode>() ? FOLLOW : EXCLUDE; });
  for (const auto &cnode : cnodes) {
    if (common::AnfAlgo::IsGraphKernel(cnode) && common::AnfAlgo::IsDynamicShape(cnode)) {
      auto func_graph = GetCNodeFuncGraph(cnode);
      MS_EXCEPTION_IF_NULL(func_graph);
      if (func_graph->has_attr(kAttrSymbolEngine)) {
        MS_LOG(DEBUG) << "Set infershape functor for cnode: " << cnode->fullname_with_scope();
        common::AnfAlgo::SetNodeAttrSafely("infer_shape_functor",
                                           std::make_shared<SymbolEngineInfer>("symbol_engine_infer_functor"), cnode);
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
