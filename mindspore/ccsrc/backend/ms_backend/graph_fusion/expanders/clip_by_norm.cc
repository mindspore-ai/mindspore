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
#include "ir/tensor_new.h"
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class ClipByNorm : public OpDesc {
 public:
  ClipByNorm() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~ClipByNorm() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x0 = inputs[0];
    const auto &input_x1 = inputs[1];
    auto dim = input_x0->shape.size();
    bool need_cast = (input_x0->type != kNumberTypeFloat32);

    auto square = gb.Mul(input_x0, input_x0);
    ShapeVector axis;
    auto axis_value = attrs_["axis"];
    if (axis_value->isa<ValueSequence>()) {
      axis = GetValue<std::vector<int64_t>>(axis_value);
      if (axis.empty()) {  // reduce_sum for all dimensions
        for (size_t i = 0; i < dim; ++i) {
          (void)axis.emplace_back(i);
        }
      }
    } else if (axis_value->isa<Int64Imm>()) {
      (void)axis.emplace_back(GetValue<int64_t>(axis_value));
    }
    auto reduce_sum = gb.ReduceSum(square, axis, true);
    if (need_cast) {
      reduce_sum = gb.Cast(reduce_sum, kNumberTypeFloat32);
    }

    size_t data_len = LongToSize(std::accumulate(reduce_sum->shape.begin(), reduce_sum->shape.end(),
                                                 static_cast<int64_t>(1), std::multiplies<int64_t>()));

    std::vector<float> tensor_data0(data_len, 0.0);
    std::vector<float> tensor_data1(data_len, 1.0);

    auto tensor_zero =
      tensor::from_buffer(kNumberTypeFloat32, reduce_sum->shape, tensor_data0.data(), kNumberTypeFloat32);
    auto tensor_one =
      tensor::from_buffer(kNumberTypeFloat32, reduce_sum->shape, tensor_data1.data(), kNumberTypeFloat32);

    auto greater = gb.Greater(reduce_sum, gb.Value(tensor_zero));
    auto safe_reduce_sum = gb.Select(greater, reduce_sum, gb.Value(tensor_one));
    auto sqrt = gb.Sqrt(safe_reduce_sum);
    auto safe_sqrt = gb.Select(greater, sqrt, reduce_sum);
    NodePtr inp_x_cast = input_x0;
    if (need_cast) {
      inp_x_cast = gb.Cast(input_x0, kNumberTypeFloat32);
    }
    NodePtr clip_norm_cast = input_x1;
    if (need_cast) {
      clip_norm_cast = gb.Cast(input_x1, kNumberTypeFloat32);
    }
    auto mul = gb.Mul(inp_x_cast, clip_norm_cast);
    auto max = gb.Emit("Maximum", {clip_norm_cast, safe_sqrt});
    auto result = gb.Div(mul, max);

    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("ClipByNorm", ClipByNorm);
}  // namespace mindspore::graphkernel::expanders
