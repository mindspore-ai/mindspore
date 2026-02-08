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
#include "backend/ms_backend/graph_fusion/multi_symbol_engine.h"
#include <utility>
#include <vector>
#include <memory>
#include <map>
#include "abstract/symbolic_shape/utils.h"
#include "abstract/symbolic_shape/int_symbol.h"

namespace mindspore {
namespace graphkernel {
namespace symshape {
using mindspore::symshape::CloneAbstractIfSymbolExists;

void MultiSymbolEngine::SaveInputParaMap(std::map<SymbolPtr, SymbolPtr> *input_para_map, const SymbolPtr &inp,
                                         const SymbolPtr &para) {
  if (inp->tid() != para->tid()) {
    MS_LOG(WARNING) << "The type of input and para are not match, " << inp->type_name() << " vs " << para->type_name();
    return;
  }
  (*input_para_map)[inp] = para;
}

ListSymbolPtr MultiSymbolEngine::BuildShapeWithInputHint(const AbstractBasePtr &para_abs,
                                                         const std::vector<ListSymbolPtr> &inputs,
                                                         std::map<SymbolPtr, SymbolPtr> *input_para_map) {
  // only support TensorShape, that input is int-list symbol.
  if (!para_abs->GetShape()->isa<abstract::TensorShape>()) {
    return nullptr;
  }
  auto cur_shape = inputs.back();
  for (auto &inp_para : *input_para_map) {
    if (cur_shape->EqualsTo(inp_para.first)) {
      return inp_para.second->as_sptr<ListSymbol>();
    }
  }
  if (cur_shape->is_dyn_len()) {
    return ListSymbol::Make();
  }
  SymbolPtrList para_shape;
  para_shape.reserve(cur_shape->size());
  for (auto &cur_item : cur_shape->symbols()) {
    if (cur_item->is<IntSymbol>() && cur_item->HasData()) {
      (void)para_shape.emplace_back(cur_item);
      continue;
    }
    SymbolPtr para_item = nullptr;
    for (auto &inp_para : *input_para_map) {
      if (cur_item->EqualsTo(inp_para.first)) {
        para_item = inp_para.second;
        break;
      }
    }
    if (para_item == nullptr) {
      (void)para_shape.emplace_back(IntSymbol::Make());
      SaveInputParaMap(input_para_map, cur_item, para_shape.back());
    } else {
      (void)para_shape.emplace_back(std::move(para_item));
    }
  }
  return ListSymbol::Make(std::move(para_shape));
}

// set symbol info for subgraph's parameters, according to the outer cnode's input symbol info.
void MultiSymbolEngine::GenInputSymbols(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index) {
  std::vector<ListSymbolPtr> input_symbolic_shapes;
  std::map<SymbolPtr, SymbolPtr> input_para_map;
  input_symbolic_shapes.reserve(cnode->size());
  for (size_t i = 0; i < sub_fg->parameters().size(); i++) {
    auto inp_abs = cnode->input(i + begin_input_index)->abstract();
    MS_EXCEPTION_IF_NULL(inp_abs);
    auto para_abs = CloneAbstractIfSymbolExists(sub_fg->parameters()[i]);
    MS_EXCEPTION_IF_NULL(para_abs);
    (void)input_symbolic_shapes.emplace_back(inp_abs->GetSymbolicShape());
    if (input_symbolic_shapes.back() != nullptr) {
      auto s = BuildShapeWithInputHint(para_abs, input_symbolic_shapes, &input_para_map);
      if (s == nullptr || !s->is<ListSymbol>()) {
        s = para_abs->GetShape()->BuildSymbolicShape();
      }
      para_abs->SetSymbolicShape(s->as_sptr<ListSymbol>());
      SaveInputParaMap(&input_para_map, input_symbolic_shapes.back(), s);
    }
    if (inp_abs->GetSymbolicValue() != nullptr) {
      para_abs->SetSymbolicValue(mindspore::symshape::BuildSymbolicValue(para_abs));
    }
  }
}

void MultiSymbolEngine::Build(const FuncGraphPtr &func_graph) {
  auto engine = std::make_shared<MultiSymbolEngine>(func_graph);
  func_graph->set_symbol_engine(engine);
  engine->PreBuild();
  engine->BuildImpl();
}

void MultiSymbolEngine::BuildSubEngine(const AnfNodePtr &node) {
  auto sub_fg = GetCNodeFuncGraph(node);
  MS_EXCEPTION_IF_NULL(sub_fg);
  auto engine = std::make_shared<MultiSymbolEngine>(sub_fg);
  sub_fg->set_symbol_engine(engine);
  engine->PreBuild();
  auto main_engine = node->func_graph()->symbol_engine();
  if (main_engine != nullptr && main_engine->isa<MultiSymbolEngine>()) {
    main_engine->cast_ptr<MultiSymbolEngine>()->BuildSubgraphImpl(node->cast<CNodePtr>(), sub_fg, 1);
  } else {
    engine->BuildImpl();
  }
}

void MultiSymbolEngine::PreBuildQuerySubgraphDependStatus(const CNodePtr &cnode, const FuncGraphPtr &sub_fg,
                                                          size_t begin_input_index) {
  auto sub_engine = std::make_shared<MultiSymbolEngine>(sub_fg);
  sub_fg->set_symbol_engine(sub_engine);
  sub_engine->depend_status_map_[sub_fg->output()] = this->depend_status_map_[cnode];
  sub_engine->PreBuild();

  for (auto &param : sub_fg->parameters()) {
    auto &cnode_input_depend_status = this->depend_status_map_[cnode->input(begin_input_index++)];
    auto depend_status = sub_engine->depend_status_map_[param];
    if (depend_status.shape) {
      cnode_input_depend_status.shape = true;
    }
    if (depend_status.value) {
      cnode_input_depend_status.value = true;
    }
  }
}

void MultiSymbolEngine::BuildSubgraphImpl(const CNodePtr &cnode, const FuncGraphPtr &sub_fg, size_t begin_input_index) {
  MS_EXCEPTION_IF_NULL(sub_fg);
  MS_LOG(DEBUG) << "Build subgraph " << sub_fg->ToString() << " of node " << cnode->fullname_with_scope();

  MS_EXCEPTION_IF_NULL(sub_fg->symbol_engine());
  auto sub_engine = sub_fg->symbol_engine()->cast_ptr<MultiSymbolEngine>();
  MS_EXCEPTION_IF_NULL(sub_engine);
  GenInputSymbols(cnode, sub_fg, begin_input_index);

  sub_engine->BuildImpl();

  auto out_abs = sub_fg->output()->abstract();
  MS_EXCEPTION_IF_NULL(out_abs);
  auto cnode_abs = CloneAbstractIfSymbolExists(cnode);
  MS_EXCEPTION_IF_NULL(cnode_abs);
  cnode_abs->SetSymbolicShape(out_abs->GetSymbolicShape());
  cnode_abs->SetSymbolicValue(out_abs->GetSymbolicValue());
}
}  // namespace symshape
}  // namespace graphkernel
}  // namespace mindspore
