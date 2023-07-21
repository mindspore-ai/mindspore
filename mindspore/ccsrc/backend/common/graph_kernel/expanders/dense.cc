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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class Dense : public OpDesc {
 public:
  Dense() {
    std::initializer_list<std::string> attrs{"has_bias"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Dense() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const int64_t kTwo = 2;
    auto x = inputs[0];
    auto w = inputs[1];
    auto x_shape = x->shape;
    if (x_shape.size() != kTwo) {
      ShapeVector x_2d_shape = {-1, *(x_shape.end() - 1)};
      x = gb.Reshape(x, x_2d_shape);
    }
    x = gb.MatMul(x, w, x->type, false, true);
    auto has_bias = GetValue<bool>(attrs_["has_bias"]);
    if (has_bias) {
      auto b = inputs[2];
      x = gb.Emit("BiasAdd", {x, b}, {{"format", MakeValue("NCHW")}});
    }
    if (x_shape.size() != kTwo) {
      ShapeVector out_shape{x_shape.begin(), x_shape.end() - 1};
      out_shape.push_back(*(x->shape.end() - 1));
      x = gb.Reshape(x, out_shape);
    }

    return {x};
  }
};
EXPANDER_OP_DESC_REGISTER("Dense", Dense);
}  // namespace mindspore::graphkernel::expanders
