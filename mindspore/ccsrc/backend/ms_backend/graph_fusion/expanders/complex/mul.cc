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
class CMul : public OpDesc {
 public:
  CMul() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
  }
  ~CMul() = default;

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
      auto x_real_mul_y_real = gb.Mul(x_real, y_real);
      auto x_real_mul_y_imag = gb.Mul(x_real, y_imag);
      auto x_imag_mul_y_real = gb.Mul(x_imag, y_real);
      auto x_imag_mul_y_imag = gb.Mul(x_imag, y_imag);
      auto result_real = gb.Sub(x_real_mul_y_real, x_imag_mul_y_imag);
      auto result_imag = gb.Add(x_real_mul_y_imag, x_imag_mul_y_real);
      result = gb.Complex(result_real, result_imag);
    } else if (x->type == TypeId::kNumberTypeComplex64 || x->type == TypeId::kNumberTypeComplex128) {
      auto x_real = gb.CReal(x);
      auto x_imag = gb.CImag(x);
      auto result_real = gb.Mul(x_real, y);
      auto result_imag = gb.Mul(x_imag, y);
      result = gb.Complex(result_real, result_imag);
    } else if (y->type == TypeId::kNumberTypeComplex128 || y->type == TypeId::kNumberTypeComplex64) {
      auto y_real = gb.CReal(y);
      auto y_imag = gb.CImag(y);
      auto result_real = gb.Mul(x, y_real);
      auto result_imag = gb.Mul(x, y_imag);
      result = gb.Complex(result_real, result_imag);
    }
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("CMul", CMul);
}  // namespace mindspore::graphkernel::expanders
