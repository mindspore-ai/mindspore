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
class CAbs : public OpDesc {
 public:
  CAbs() = default;
  ~CAbs() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];

    auto x_real = gb.CReal(x);
    auto x_imag = gb.CImag(x);
    auto square_x_real = gb.Mul(x_real, x_real);
    auto square_x_imag = gb.Mul(x_imag, x_imag);
    auto square_sum = gb.Add(square_x_real, square_x_imag);
    auto result = gb.Sqrt(square_sum);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("CAbs", CAbs);
}  // namespace mindspore::graphkernel::expanders
