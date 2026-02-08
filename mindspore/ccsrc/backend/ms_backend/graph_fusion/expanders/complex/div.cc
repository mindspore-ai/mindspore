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

#include <memory>
#include <vector>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
// Complex Div
class CDiv : public OpDesc {
 public:
  CDiv() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
  }
  ~CDiv() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    const auto &y = inputs[1];
    NodePtr result;
    if (x->type == y->type) {
      auto x_real = gb.CReal(x);
      auto y_real = gb.CReal(y);
      auto x_imag = gb.CImag(x);
      auto y_imag = gb.CImag(y);
      auto square_y_real = gb.Mul(y_real, y_real);
      auto square_y_imag = gb.Mul(y_imag, y_imag);
      auto final_denominator = gb.Add(square_y_real, square_y_imag);
      auto x_real_mul_y_real = gb.Mul(x_real, y_real);
      auto x_real_mul_y_imag = gb.Mul(x_real, y_imag);
      auto x_imag_mul_y_real = gb.Mul(x_imag, y_real);
      auto x_imag_mul_y_imag = gb.Mul(x_imag, y_imag);
      auto final_numerator_real = gb.Add(x_real_mul_y_real, x_imag_mul_y_imag);
      auto final_numerator_imag = gb.Sub(x_imag_mul_y_real, x_real_mul_y_imag);
      auto result_real = gb.Div(final_numerator_real, final_denominator);
      auto result_imag = gb.Div(final_numerator_imag, final_denominator);
      result = gb.Complex(result_real, result_imag);
    } else if (x->type == TypeId::kNumberTypeComplex64 || x->type == TypeId::kNumberTypeComplex128) {
      auto x_real = gb.CReal(x);
      auto x_imag = gb.CImag(x);
      auto result_real = gb.Div(x_real, y);
      auto result_imag = gb.Div(x_imag, y);
      result = gb.Complex(result_real, result_imag);
    } else if (y->type == TypeId::kNumberTypeComplex128 || y->type == TypeId::kNumberTypeComplex64) {
      auto y_real = gb.CReal(y);
      auto y_imag = gb.CImag(y);
      auto neg_y_imag = gb.Neg(y_imag);
      auto square_y_real = gb.Mul(y_real, y_real);
      auto square_y_imag = gb.Mul(y_imag, y_imag);
      auto final_denominator = gb.Add(square_y_real, square_y_imag);
      auto x_mul_y_real = gb.Mul(x, y_real);
      auto x_mul_neg_y_imag = gb.Mul(x, neg_y_imag);
      auto result_real = gb.Div(x_mul_y_real, final_denominator);
      auto result_imag = gb.Div(x_mul_neg_y_imag, final_denominator);
      result = gb.Complex(result_real, result_imag);
    }
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("CDiv", CDiv);
EXPANDER_OP_DESC_REGISTER("CRealDiv", CDiv);
}  // namespace mindspore::graphkernel::expanders
