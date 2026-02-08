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
class FloatStatus : public OpDesc {
 public:
  FloatStatus() = default;
  ~FloatStatus() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    auto input_x = inputs[0];
    if (input_x->type == kNumberTypeBFloat16) {
      input_x = gb.Cast(input_x, kNumberTypeFloat32);
    }
    if (processor_ == "aicore") {
      auto res1 = gb.Emit("IsFinite", {input_x});
      auto res2 = gb.Emit("LogicalNot", {res1});
      auto res4 = gb.Emit("ElemAny", {res2},
                          {{"dst_type", kFloat32}, {"empty_shape", MakeValue(outputs_info_[0].shape.empty())}});
      return {res4};
    }
    auto res1 = gb.IsInf(input_x);
    auto res2 = gb.IsNan(input_x);
    auto res3 = gb.LogicalOr(res1, res2);
    auto res4 = gb.Emit("ElemAny", {res3}, {{"dst_type", kFloat32}});
    return {res4};
  }
};
EXPANDER_OP_DESC_REGISTER("FloatStatus", FloatStatus);
}  // namespace mindspore::graphkernel::expanders
