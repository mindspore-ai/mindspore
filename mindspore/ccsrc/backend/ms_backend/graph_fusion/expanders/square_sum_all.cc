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
class SquareSumAll : public OpDesc {
 public:
  SquareSumAll() {}
  ~SquareSumAll() = default;

 protected:
  bool CheckInputs() override {
    auto input_num = inputs_info_.size();
    if (input_num != kIndex2) {
      MS_LOG(INFO) << "For 'SquareSumAll', the inputs number should be 2, but got " << input_num;
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    const auto &y = inputs[1];

    ShapeVector axis;
    for (size_t i = 0; i < x->shape.size(); ++i) {
      axis.push_back(SizeToLong(i));
    }

    auto square_x = gb.Mul(x, x);
    auto square_y = gb.Mul(y, y);
    auto output_x = gb.ReduceSum(square_x, axis);
    auto output_y = gb.ReduceSum(square_y, axis);

    return {output_x, output_y};
  }
};
EXPANDER_OP_DESC_REGISTER("SquareSumAll", SquareSumAll);
}  // namespace mindspore::graphkernel::expanders
