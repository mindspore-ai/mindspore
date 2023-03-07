/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class MatMulFusion : public OpDesc {
 public:
  MatMulFusion() {
    std::set<int64_t> activation_types = {ActivationType::NO_ACTIVATION, ActivationType::RELU, ActivationType::SIGMOID};
    (void)validators_.emplace_back(std::make_unique<CheckActivationType>(activation_types));
  }
  ~MatMulFusion() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const size_t has_bias_input_size = 3;
    auto a = inputs[0];
    auto b = inputs[1];
    auto bias = (inputs.size() == has_bias_input_size) ? inputs[2] : nullptr;
    auto transpose_a = (attrs_.count("transpose_a") != 0) ? attrs_["transpose_a"] : MakeValue(false);
    auto transpose_b = (attrs_.count("transpose_b") != 0) ? attrs_["transpose_b"] : MakeValue(false);
    auto pack_b = (attrs_.count("pack_b") != 0) ? attrs_["pack_b"] : MakeValue(false);
    if (b->shape.size() < a->shape.size()) {
      ShapeVector new_shape(a->shape.size() - b->shape.size(), 1);
      (void)new_shape.insert(new_shape.end(), b->shape.cbegin(), b->shape.cend());
      b = gb.Reshape(b, new_shape);
    } else if (a->shape.size() < b->shape.size()) {
      ShapeVector new_shape(b->shape.size() - a->shape.size(), 1);
      (void)new_shape.insert(new_shape.end(), a->shape.cbegin(), a->shape.cend());
      a = gb.Reshape(a, new_shape);
    }
    auto matmul =
      gb.Emit("MatMul", {a, b},
              {{"transpose_a", transpose_a}, {"transpose_b", transpose_b}, {"dst_type", kFloat32}, {"pack_b", pack_b}});
    if (bias != nullptr) {
      matmul = gb.Add(matmul, bias);
    }
    if (attrs_.find("activation_type") != attrs_.end()) {
      auto act_type = GetValue<int64_t>(attrs_["activation_type"]);
      return {GetActivationExpander(gb, {matmul}, act_type)};
    } else {
      return {matmul};
    }
  }
};
EXPANDER_OP_DESC_REGISTER("MatMulFusion", MatMulFusion);
}  // namespace mindspore::graphkernel::expanders
