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
#include "backend/common/graph_kernel/adapter/symbol_engine_builder.h"
#include <memory>
#include <map>
#include <functional>
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/symbol_engine/multi_symbol_engine.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "ops/auto_generate/gen_ops_primitive.h"

namespace mindspore::graphkernel {
namespace {
using RectifyFunc = std::function<void(const CNodePtr &)>;

void RectifySiLUGrad(const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimMul)) {
    auto x0_shape = GkUtils::GetOutputSymbolicShape(node->input(kIndex1), 0);
    auto x1_shape = GkUtils::GetOutputSymbolicShape(node->input(kIndex2), 0);
    auto out_shape = GkUtils::GetOutputSymbolicShape(node, 0);
    if (x0_shape != nullptr && x1_shape != nullptr && out_shape != nullptr && x0_shape->size() == x1_shape->size()) {
      for (size_t i = 0; i < x0_shape->size(); ++i) {
        auto sh0 = x0_shape->item(i);
        auto sh1 = x1_shape->item(i);
        auto sho = out_shape->item(i);
        if (sh0 == nullptr || sh1 == nullptr || sho == nullptr || sh0->EqualsTo(sh1)) {
          continue;
        }
        MS_LOG(DEBUG) << "Set symbols equal: " << sh0->ToString() << ", " << sh1->ToString() << ", " << sho->ToString()
                      << " node: " << node->fullname_with_scope();
        sh0->as<IntSymbol>()->SetEqual(sh1->as_sptr<IntSymbol>());
        sh0->as<IntSymbol>()->SetEqual(sho->as_sptr<IntSymbol>());
      }
    }
  }
}

void RectifySymbol(const FuncGraphPtr &func_graph) {
  static std::map<std::string, RectifyFunc> funcs{
    {"SiLUGrad", RectifySiLUGrad},
  };

  MS_EXCEPTION_IF_NULL(func_graph);
  auto nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    if (node == nullptr || !common::AnfAlgo::IsGraphKernel(node) || !common::AnfAlgo::IsDynamicShape(node)) {
      continue;
    }
    auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_graph);
    auto inner_nodes = TopoSort(sub_graph->get_return());
    for (const auto &n : inner_nodes) {
      if (n == nullptr || !n->isa<CNode>()) {
        continue;
      }
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto expand_from = cnode->GetAttr(kAttrExpandFrom);
      if (expand_from == nullptr) {
        continue;
      }
      auto expand_op_name = GetValue<std::string>(expand_from);
      auto iter = funcs.find(expand_op_name);
      if (iter != funcs.end()) {
        iter->second(cnode);
      }
    }
  }
}
}  // namespace

bool SymbolEngineBuilder::Run(const FuncGraphPtr &func_graph) {
  if (!common::AnfAlgo::IsDynamicGraph(func_graph)) {
    return false;
  }
  if (multi_engine_) {
    symshape::MultiSymbolEngine::Build(func_graph);
  } else {
    symshape::SymbolEngineImpl::Build(func_graph);
  }
  RectifySymbol(func_graph);
  return true;
}
}  // namespace mindspore::graphkernel
