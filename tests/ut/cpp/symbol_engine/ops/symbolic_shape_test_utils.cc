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
#include "abstract/symbolic_shape/utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "mindspore/core/abstract/symbolic_shape/symbol_utils.h"

namespace mindspore::symshape::test {
void TestSymbolEngine::SaveIR(const FuncGraphPtr &fg, const std::string &name) {
  if (common::GetEnv("MS_DEV_SAVE_GRAPHS") == "") {
    return;
  }
  MS_EXCEPTION_IF_NULL(fg);
  auto current_test_info = testing::UnitTest::GetInstance()->current_test_info();
  std::string case_name = current_test_info->test_case_name();
  std::string cur_name = current_test_info->name();
  auto fname = case_name + "." + cur_name + ".";
  if (name == "") {
    fname += "fg.ir";
  } else {
    fname += name + ".ir";
  }
  DumpIR(fname, fg, true);
}

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

void OperationTestHelper::Infer(const std::vector<std::pair<ListSymbolPtr, ShapeVector>> &real_shape) {
  DumpText();
  for (auto &[listsym, shape] : real_shape) {
    listsym->Update(ShapeVector2Symbol(shape));
  }
  for (auto &op : ops_list_) {
    op->Run();
  }
}

void OperationTestHelper::DumpText() {
  if (common::GetEnv("MS_UT_DUMP_SYMBOL") != "on") {
    return;
  }
  auto current_test_info = testing::UnitTest::GetInstance()->current_test_info();
  std::string case_name = current_test_info->test_case_name();
  std::string cur_name = current_test_info->name();
  std::cout << "==== Ops in " << case_name << "." << cur_name << " ====" << std::endl;
  for (auto &op : ops_list_) {
    std::cout << op->DumpText();
  }
  std::cout << "=====================\n" << std::endl;
}
}  // namespace mindspore::symshape::test
