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
#ifndef MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_CAST_H_
#define MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_CAST_H_
#include "mindspore/ops/infer/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
template <typename DT, typename = std::enable_if_t<std::is_same_v<IntSymbol, DT> || std::is_same_v<FloatSymbol, DT> ||
                                                   std::is_same_v<BoolSymbol, DT>>>
class OPS_API ScalarCast : public InferValueOp {
 public:
  using DS = typename DT::elem_type;
  using InferValueOp::InferValueOp;
  explicit ScalarCast(const SymbolPtr &s) : InferValueOp({s}) {}
  ~ScalarCast() override = default;
  MS_DECLARE_PARENT(ScalarCast, InferValueOp)

 protected:
  SymbolPtr Eval() override {
    auto x = input(0);
    if (x->template is<DT>()) {
      DoNotEvalOnRun();
      return x;
    }
    if (x->HasData()) {
      if (auto v = x->template as_noexcept<IntSymbol>(); v != nullptr) {
        return DT::Make(static_cast<DS>(v->value()));
      } else if (auto v = x->template as_noexcept<FloatSymbol>(); v != nullptr) {
        return DT::Make(static_cast<DS>(v->value()));
      } else if (auto v = x->template as_noexcept<BoolSymbol>(); v != nullptr) {
        return DT::Make(static_cast<DS>(v->value()));
      }
      MS_LOG(INTERNAL_EXCEPTION) << "ScalarCast only int/float/bool symbol, but got " << x->type_name();
    }
    return DT::Make(shared_from_this());
  }
  void EvalOnRun() override {
    auto x = input(0);
    if (auto v = x->template as_noexcept<IntSymbol>(); v != nullptr) {
      return output_as<DT>()->SetValue(static_cast<DS>(v->value()));
    } else if (auto v = x->template as_noexcept<FloatSymbol>(); v != nullptr) {
      return output_as<DT>()->SetValue(static_cast<DS>(v->value()));
    } else if (auto v = x->template as_noexcept<BoolSymbol>(); v != nullptr) {
      return output_as<DT>()->SetValue(static_cast<DS>(v->value()));
    }
  }
};
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SYMBOL_OPS_IMPL_SCALAR_CAST_H_
