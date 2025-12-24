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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_DIV_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_DIV_H_

#include <cmath>
#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
class OPS_API ScalarDiv : public ScalarIntOp {
 public:
  using ScalarIntOp::ScalarIntOp;
  ScalarDiv(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarIntOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarDiv, ScalarIntOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(DivWithCheck(AsInt(input(0)), AsInt(input(1)))); }
  inline int64_t DivWithCheck(int64_t x, int64_t y) const {
    if (y == 0) {
      MS_EXCEPTION(ValueError) << "For operation 'ScalarDiv', the denominator can not be zero.";
    }
    if (x % y != 0) {
      MS_LOG(EXCEPTION) << "For operation 'ScalarDiv', the 'x' should be divisible by 'y', but got " << x << "/" << y;
    }
    int64_t minus_one = -1;
    if (x == INT64_MIN && y == minus_one) {
      MS_EXCEPTION(ValueError) << "For operation 'ScalarDiv', the denominator can not be -1 when the numerator is "
                                  "INT64_MIN, or else overflow will occur.";
    }
    return x / y;
  }
  void UpdateMathInfo() override;
};

class OPS_API ScalarFloorDiv : public ScalarIntOp {
 public:
  using ScalarIntOp::ScalarIntOp;
  ScalarFloorDiv(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarIntOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarFloorDiv, ScalarIntOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(FloorDiv(AsInt(input(0)), AsInt(input(1)))); }
  inline int64_t FloorDiv(int64_t x, int64_t y) const {
    if (y == 0) {
      MS_EXCEPTION(ValueError) << "For operation 'ScalarFloorDiv', the denominator can not be zero.";
    }
    return DoubleToLong(std::floor(LongToDouble(x) / LongToDouble(y)));
  }
  void UpdateMathInfo() override;
};

class OPS_API ScalarCeilDiv : public ScalarIntOp {
 public:
  using ScalarIntOp::ScalarIntOp;
  ScalarCeilDiv(const SymbolPtr &lhs, const SymbolPtr &rhs) : ScalarIntOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarCeilDiv, ScalarIntOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<IntSymbol>()->SetValue(CeilDiv(AsInt(input(0)), AsInt(input(1)))); }
  inline int64_t CeilDiv(int64_t x, int64_t y) const {
    if (y == 0) {
      MS_EXCEPTION(ValueError) << "For operation 'ScalarCeilDiv', the denominator can not be zero.";
    }
    return DoubleToLong(std::ceil(LongToDouble(x) / LongToDouble(y)));
  }
};

class OPS_API ScalarRealDiv : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  ScalarRealDiv(const SymbolPtr &lhs, const SymbolPtr &rhs) : InferValueOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarRealDiv, InferValueOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override { output_as<FloatSymbol>()->SetValue(DivWithCheck(AsInt(input(0)), AsInt(input(1)))); }
  inline double DivWithCheck(int64_t x, int64_t y) const {
    if (y == 0) {
      MS_LOG(EXCEPTION) << "For ScalarRealDiv, the denominator can not be zero.";
    }
    return LongToDouble(x) / LongToDouble(y);
  }
};
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_DIV_H_
