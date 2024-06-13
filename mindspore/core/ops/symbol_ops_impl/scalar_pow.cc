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
#include "mindspore/core/ops/symbol_ops_impl/common.h"
#include "mindspore/core/ops/symbol_ops_impl/scalar_cast.h"

namespace mindspore {
namespace symshape {
namespace ops {
class MS_CORE_API ScalarPow : public InferValueOp {
 public:
  using InferValueOp::InferValueOp;
  ScalarPow(const SymbolPtr &lhs, const SymbolPtr &rhs) : InferValueOp({lhs, rhs}) {}
  MS_DECLARE_PARENT(ScalarPow, InferValueOp)

 protected:
  SymbolPtr Eval() override;
  void EvalOnRun() override;
  void UnifyInputsType() {
    x_ = input(0);
    y_ = input(1);
    if (x_->is<FloatSymbol>() || y_->is<FloatSymbol>()) {
      x_ = Emit(std::make_shared<ScalarCast<FloatSymbol>>(x_));
      y_ = Emit(std::make_shared<ScalarCast<FloatSymbol>>(y_));
    } else if (x_->is<IntSymbol>() || y_->is<IntSymbol>()) {
      x_ = Emit(std::make_shared<ScalarCast<IntSymbol>>(x_));
      y_ = Emit(std::make_shared<ScalarCast<IntSymbol>>(y_));
    } else if (!x_->is<BoolSymbol>() && !y_->is<BoolSymbol>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Scalar ops only support float/int/bool symbol, but got " << x_->type_name()
                                 << " and " << y_->type_name();
    }
  }
  template <typename T, typename S = typename T::elem_type>
  SymbolPtr Process() {
    const S v0 = 0;
    const S v1 = 1;
    if (y_->HasData()) {
      auto y = y_->as<T>()->value();
      if (x_->HasData()) {
        return T::Make(static_cast<S>(std::pow(static_cast<double>(x_->as<T>()->value()), static_cast<double>(y))));
      }
      if (y == v0) {
        return T::Make(v1);
      }
      if (y == v1) {
        return x_;
      }
    }
    return T::Make(shared_from_this());
  }

  SymbolPtr x_{nullptr};
  SymbolPtr y_{nullptr};
};

SymbolPtr ScalarPow::Eval() {
  UnifyInputsType();
  if (x_->is<FloatSymbol>()) {
    return Process<FloatSymbol>();
  } else if (x_->is<IntSymbol>()) {
    return Process<IntSymbol>();
  }
  return Process<BoolSymbol>();
}

void ScalarPow::EvalOnRun() {
  if (output_->is<FloatSymbol>()) {
    output_as<FloatSymbol>()->SetValue(std::pow(x_->as<FloatSymbol>()->value(), y_->as<FloatSymbol>()->value()));
  } else if (output_->is<IntSymbol>()) {
    output_as<IntSymbol>()->SetValue(static_cast<int64_t>(
      std::pow(static_cast<double>(x_->as<IntSymbol>()->value()), static_cast<double>(y_->as<IntSymbol>()->value()))));
  } else {
    output_as<BoolSymbol>()->SetValue(static_cast<bool>(std::pow(static_cast<double>(x_->as<BoolSymbol>()->value()),
                                                                 static_cast<double>(y_->as<BoolSymbol>()->value()))));
  }
}

REG_SYMBOL_OP_BUILDER("ScalarPow").SetValueDependN<DependOn::kValue, 2>().SetValueFuncWith<ScalarPow>();
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
