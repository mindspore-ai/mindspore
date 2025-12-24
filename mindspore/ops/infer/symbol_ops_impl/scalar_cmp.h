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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_CMP_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_CMP_H_

#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API ScalarCmpOp : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  ScalarCmpOp(const SymbolPtr &a, const SymbolPtr &b) : InferValueOp({a, b}) {}
  ~ScalarCmpOp() override = default;
  MS_DECLARE_PARENT(ScalarCmpOp, InferValueOp)
 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override {
    output_as<BoolSymbol>()->SetValue(Compare(input_as<IntSymbol>(0), input_as<IntSymbol>(1)));
  }
  virtual bool Compare(const IntSymbol *a, const IntSymbol *b) const = 0;
};

class OPS_API ScalarEq final : public ScalarCmpOp {
 public:
  using ScalarCmpOp::ScalarCmpOp;
  MS_DECLARE_PARENT(ScalarEq, ScalarCmpOp)
 protected:
  bool Compare(const IntSymbol *a, const IntSymbol *b) const override { return *a == *b; }
};

class OPS_API ScalarGt final : public ScalarCmpOp {
 public:
  using ScalarCmpOp::ScalarCmpOp;
  MS_DECLARE_PARENT(ScalarGt, ScalarCmpOp)
 protected:
  bool Compare(const IntSymbol *a, const IntSymbol *b) const override { return *a > *b; }
};

class OPS_API ScalarGe final : public ScalarCmpOp {
 public:
  using ScalarCmpOp::ScalarCmpOp;
  MS_DECLARE_PARENT(ScalarGe, ScalarCmpOp)
 protected:
  bool Compare(const IntSymbol *a, const IntSymbol *b) const override { return *a >= *b; }
};

class OPS_API ScalarLt final : public ScalarCmpOp {
 public:
  using ScalarCmpOp::ScalarCmpOp;
  MS_DECLARE_PARENT(ScalarLt, ScalarCmpOp)
 protected:
  bool Compare(const IntSymbol *a, const IntSymbol *b) const override { return *a < *b; }
};

class OPS_API ScalarLe final : public ScalarCmpOp {
 public:
  using ScalarCmpOp::ScalarCmpOp;
  MS_DECLARE_PARENT(ScalarLe, ScalarCmpOp)
 protected:
  bool Compare(const IntSymbol *a, const IntSymbol *b) const override { return *a <= *b; }
};
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_CMP_H_
