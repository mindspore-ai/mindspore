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

#ifndef UT_CPP_SYMBOL_ENGINE_OPS_SYMBOLIC_SHAPE_TEST_UTILS_H_
#define UT_CPP_SYMBOL_ENGINE_OPS_SYMBOLIC_SHAPE_TEST_UTILS_H_

#include "abstract/symbolic_shape/symbol.h"
#include "mindspore/ccsrc/include/utils/symbol_engine/symbol_engine_impl.h"
#include "abstract/symbolic_shape/int_symbol.h"
#include "common/common_test.h"
#include "utils/ms_context.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/jit/ps/parse/data_converter.h"
#include "abstract/symbolic_shape/symbol_info.h"

namespace mindspore::symshape::test {
using IntSymbolInfo = symshape::SymbolInfo;
class SymbolEngineImplTestHelper {
 public:
  // Initialize a symbol engine, but not build symbolic shape of ops.
  void InitSymbolEngine(const FuncGraphPtr &fg);
  // Initialize a symbol engine, and then build symbolic shape of ops.
  void BuildSymbolEngine(const FuncGraphPtr &fg);

  ListSymbolPtr SetSymbolicShapeInfo(const AnfNodePtr &node, const SymbolInfoList &symbol_info);
  ListSymbolPtr BuildSymbolicShape(const CNodePtr &cnode);
  SymbolPtr BuildSymbolicValue(const CNodePtr &cnode);
  bool SupportInfer() { return symbol_engine_->SupportInfer(); }
  bool Infer(const AbstractBasePtrList &inputs) { return symbol_engine_->Infer(inputs); }

  // check symbolic_shape == digital_shape.
  bool CheckSymbolicShapeMatchesDigitalShape(const AnfNodePtr &node);

  BaseShapePtr ConvertSymbolToShape(const AnfNodePtr &node);
  ValuePtr ConvertSymbolToValue(const AnfNodePtr &node);

 protected:
  // Use a inner class to make the testsuite can visit the protected functions of SymbolEngineImpl
  class InnerSymbolEngine : public SymbolEngineImpl {
   public:
    friend class SymbolEngineImplTestHelper;
    using SymbolEngineImpl::SymbolEngineImpl;
  };
  std::shared_ptr<InnerSymbolEngine> symbol_engine_{nullptr};
  void CheckSymbolEngineExists(const std::string &fn_name) const {
    if (symbol_engine_ == nullptr) {
      MS_LOG(EXCEPTION)
        << "Please use 'InitSymbolEngine' or 'BuildSymbolEngine' to build a symbol enigne before calling " << fn_name;
    }
  }
};

class TestSymbolEngine : public UT::Common {
 public:
  void SetUp() override { helper_ = std::make_shared<SymbolEngineImplTestHelper>(); }
  void TearDown() override { helper_ = nullptr; }

  std::shared_ptr<SymbolEngineImplTestHelper> helper_{nullptr};

  void SaveIR(const FuncGraphPtr &fg, const std::string &name = "");
};

class OperationTestHelper {
 public:
  OperationTestHelper() { emitter_ = std::make_unique<OperationEmitter>(&ops_list_); }
  ~OperationTestHelper() { emitter_->Clean(); }
  const OperationEmitter &emitter() { return *emitter_; }
  SymbolPtr Emit(const OpPtr &op) { return emitter_->Emit(op); }
  void Infer(const std::vector<std::pair<ListSymbolPtr, ShapeVector>> &real_shape);

  void DumpText();
  std::unique_ptr<OperationEmitter> emitter_ = nullptr;
  OpPtrList ops_list_;
};

class TestOperation : public UT::Common {
 public:
  void SetUp() override { helper_ = std::make_shared<OperationTestHelper>(); }
  void TearDown() override {
    helper_->DumpText();
    helper_ = nullptr;
  }
  // gen const-len list
  ListSymbolPtr GenList(const SymbolPtrList &s) { return ListSymbol::Make(s); }
  // gen dyn-len list
  ListSymbolPtr GenVList() { return ListSymbol::Make(); }
  // gen const int
  IntSymbolPtr GenInt(int64_t x) { return IntSymbol::Make(x); }
  // gen variable int
  IntSymbolPtr GenVInt() { return IntSymbol::Make(); }
  // gen positive variable int
  IntSymbolPtr GenPVInt(int64_t divisor = 1, int64_t remainder = 0) {
    auto sym = IntSymbol::Make();
    sym->SetRangeMin(1);
    sym->SetDivisorRemainder(divisor, remainder);
    return sym;
  }
  std::shared_ptr<OperationTestHelper> helper_{nullptr};
};
}  // namespace mindspore::symshape::test
#endif  // UT_CPP_SYMBOL_ENGINE_OPS_SYMBOLIC_SHAPE_TEST_UTILS_H_
