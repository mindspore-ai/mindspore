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

#ifndef UT_CPP_SYMBOL_ENGINE_MATH_INFO_SYMBOL_TEST_UTILS_H_
#define UT_CPP_SYMBOL_ENGINE_MATH_INFO_SYMBOL_TEST_UTILS_H_

#include "abstract/symbolic_shape/int_symbol.h"
#include "abstract/symbolic_shape/symbol_info.h"
#include "mindspore/ccsrc/include/utils/symbol_engine/symbol_engine_impl.h"
#include "common/common_test.h"

namespace mindspore::symshape::test {
using IntSymbolInfo = symshape::SymbolInfo;
class SymbolHelper {
 public:
  SymbolPtr Emit(const OpPtr &op) { return emitter().Emit(op); }
  const OperationEmitter &emitter() {
    InitSymbolEngine();
    return *(symbol_engine_->emitter_);
  }

 protected:
  void InitSymbolEngine();
  // Use a inner class to make the testsuite can visit the protected functions of SymbolEngineImpl
  class InnerSymbolEngine : public SymbolEngineImpl {
   public:
    friend class SymbolHelper;
    using SymbolEngineImpl::SymbolEngineImpl;
  };
  std::shared_ptr<InnerSymbolEngine> symbol_engine_{nullptr};
};

class TestMathInfo : public UT::Common {
 public:
  void SetUp() override { helper_ = std::make_shared<SymbolHelper>(); }
  void TearDown() override { helper_ = nullptr; }

  std::shared_ptr<SymbolHelper> helper_{nullptr};
};
}  // namespace mindspore::symshape::test
#endif  // UT_CPP_SYMBOL_ENGINE_MATH_INFO_SYMBOL_TEST_UTILS_H_
