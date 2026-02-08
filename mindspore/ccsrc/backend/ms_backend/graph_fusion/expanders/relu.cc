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
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class ReLU : public OpDesc {
 public:
  ReLU() = default;
  ~ReLU() = default;

  static NodePtr Exec(const inner::GraphBuilder &gb, const NodePtrList &inputs) {
    auto const_zero = gb.Tensor(0.0, inputs[0]->type);
    auto result = gb.Emit("Maximum", {inputs[0], const_zero});
    return result;
  }

 protected:
  bool CheckInputs() override { return inputs_info_[0].type != kNumberTypeBool; }
  NodePtrList Expand(const NodePtrList &inputs) override { return {Exec(gb, inputs)}; }
};
EXPANDER_OP_DESC_REGISTER("ReLU", ReLU);

NodePtr ReluExpand(const inner::GraphBuilder &gb, const NodePtrList &inputs) { return ReLU::Exec(gb, inputs); }
}  // namespace mindspore::graphkernel::expanders
