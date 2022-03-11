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
#include <vector>

#include "common/graph_kernel/expanders/expander_factory.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class Sigmoid : public OpDesc {
 public:
  Sigmoid() = default;
  ~Sigmoid() = default;

  static NodePtr Exec(const inner::LiteGraph::GraphBuilder &gb, const NodePtrList &inputs) {
    const auto &input_x = inputs[0];
    auto dtype = input_x->type;
    tensor::TensorPtr data_one = std::make_shared<tensor::Tensor>(static_cast<double>(1.0), TypeIdToType(dtype));
    auto const_one = gb.Value(data_one);
    auto neg_x = gb.Emit("Neg", {input_x});
    auto exp_neg_x = gb.Emit("Exp", {neg_x});
    auto add_exp = gb.Emit("Add", {const_one, exp_neg_x});
    auto result = gb.Emit("RealDiv", {const_one, add_exp});
    return result;
  }

 protected:
  NodePtrList Expand() override { return {Exec(gb, gb.Get()->inputs())}; }
};
OP_EXPANDER_REGISTER("Sigmoid", Sigmoid);

NodePtr SigmoidExpand(const inner::LiteGraph::GraphBuilder &gb, const NodePtrList &inputs) {
  return Sigmoid::Exec(gb, inputs);
}
}  // namespace mindspore::graphkernel::expanders
