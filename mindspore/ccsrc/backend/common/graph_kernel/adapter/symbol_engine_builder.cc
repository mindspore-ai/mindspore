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

void SetSymbolShapeEqual(const std::vector<ListSymbolPtr> &shapes) {
  if (shapes.empty() || shapes[0] == nullptr) {
    return;
  }
  for (size_t i = 1; i < shapes.size(); ++i) {
    if (shapes[i] == nullptr || shapes[i]->size() != shapes[0]->size()) {
      continue;
    }
    for (size_t idx = 0; idx < shapes[0]->size(); ++idx) {
      auto a = shapes[0]->item(idx);
      auto b = shapes[i]->item(idx);
      if (a == nullptr || b == nullptr || a->EqualsTo(b)) {
        continue;
      }
      auto ia = a->as_sptr<IntSymbol>();
      auto ib = b->as_sptr<IntSymbol>();
      if (!ia->is_const()) {
        MS_LOG(DEBUG) << "Set symbols equal: " << a->ToString() << ", " << b->ToString();
        ia->SetEqual(ib);
      } else if (!ib->is_const()) {
        MS_LOG(DEBUG) << "Set symbols equal: " << a->ToString() << ", " << b->ToString();
        ib->SetEqual(ia);
      }
    }
  }
}

void SetBinaryOpSymbolShapeEqual(const CNodePtr &node) {
  auto x0_shape = GkUtils::GetOutputSymbolicShape(node->input(kIndex1), 0);
  auto x1_shape = GkUtils::GetOutputSymbolicShape(node->input(kIndex2), 0);
  auto out_shape = GkUtils::GetOutputSymbolicShape(node, 0);
  SetSymbolShapeEqual({x0_shape, x1_shape, out_shape});
}

void RectifySiLUGrad(const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimMul)) {
    SetBinaryOpSymbolShapeEqual(node);
  }
}

void RectifyAddN(const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimAdd)) {
    SetBinaryOpSymbolShapeEqual(node);
  }
}

void RectifyRmsNormGrad(const CNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimAdd) || IsPrimitiveCNode(node, prim::kPrimMul)) {
    SetBinaryOpSymbolShapeEqual(node);
  }
}

void RectifySymbol(const FuncGraphPtr &func_graph) {
  static std::map<std::string, RectifyFunc> funcs{
    {"SiLUGrad", RectifySiLUGrad}, {"AddN", RectifyAddN}, {"RmsNormGrad", RectifyRmsNormGrad}};

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
