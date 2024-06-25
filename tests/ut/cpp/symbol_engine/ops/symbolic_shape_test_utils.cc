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

#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "symbolic_shape/utils.h"

namespace mindspore::symshape::test {
void SymbolEngineImplTestHelper::InitSymbolEngine(const FuncGraphPtr &fg) {
  if (fg->symbol_engine() != nullptr) {
    CleanSymbols(fg);
  }
  symbol_engine_ = std::make_shared<InnerSymbolEngine>(fg);
  fg->set_symbol_engine(symbol_engine_);
}

void SymbolEngineImplTestHelper::BuildSymbolEngine(const FuncGraphPtr &fg) {
  if (fg->symbol_engine() != nullptr) {
    CleanSymbols(fg);
  }
  symbol_engine_ = std::make_shared<InnerSymbolEngine>(fg);
  fg->set_symbol_engine(symbol_engine_);
  symbol_engine_->PreBuild();
  symbol_engine_->BuildImpl();
}

ListSymbolPtr SymbolEngineImplTestHelper::SetSymbolicShapeInfo(const AnfNodePtr &node,
                                                               const SymbolInfoList &symbol_info) {
  auto ret = BuildSymbolicShapeBySymbolInfo({node->abstract()}, {symbol_info})[0];
  node->abstract()->SetSymbolicShape(ret);
  return ret;
}

ListSymbolPtr SymbolEngineImplTestHelper::BuildSymbolicShape(const CNodePtr &cnode) {
  CheckSymbolEngineExists("BuildSymbolicShape");
  symbol_engine_->emitter_ = std::make_unique<OperationEmitter>(&symbol_engine_->ops_);
  symbol_engine_->depend_status_map_[cnode].shape = true;
  symbol_engine_->BuildCNodeSymbol(cnode);
  symbol_engine_->depend_status_map_.erase(cnode);
  symbol_engine_->emitter_.reset(nullptr);
  return cnode->abstract()->GetSymbolicShape();
}

SymbolPtr SymbolEngineImplTestHelper::BuildSymbolicValue(const CNodePtr &cnode) {
  CheckSymbolEngineExists("BuildSymbolicValue");
  symbol_engine_->emitter_ = std::make_unique<OperationEmitter>(&symbol_engine_->ops_);
  symbol_engine_->depend_status_map_[cnode].value = true;
  symbol_engine_->BuildCNodeSymbol(cnode);
  symbol_engine_->depend_status_map_.erase(cnode);
  symbol_engine_->emitter_.reset(nullptr);
  return cnode->abstract()->GetSymbolicValue();
}

bool SymbolEngineImplTestHelper::CheckSymbolicShapeMatchesDigitalShape(const AnfNodePtr &node) {
  bool ret = (*ConvertSymbolToShape(node) == *node->abstract()->GetShape());
  if (!ret) {
    MS_LOG(ERROR) << "The digital shape is " << node->abstract()->GetShape()->ToString();
    MS_LOG(ERROR) << "The symbolic shape is " << node->abstract()->GetSymbolicShape()->ToString();
  }
  return ret;
}

BaseShapePtr SymbolEngineImplTestHelper::ConvertSymbolToShape(const AnfNodePtr &node) {
  return mindspore::symshape::QueryShape(node->abstract());
}

ValuePtr SymbolEngineImplTestHelper::ConvertSymbolToValue(const AnfNodePtr &node) {
  return mindspore::symshape::QueryValue(node->abstract());
}
}  // namespace mindspore::symshape::test
