/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class SoftplusGrad : public OpDesc {
 public:
  SoftplusGrad() {}
  ~SoftplusGrad() = default;

 protected:
  bool CheckInputs() override {
    const auto &input_x = inputs_info_[1];
    if (input_x.type != kNumberTypeFloat32 && input_x.type != kNumberTypeFloat16) {
      MS_LOG(INFO) << "In SoftplusGrad, input_x's dtype must be float16 or float32";
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_dy = inputs[0];
    const auto &input_x = inputs[1];
    auto neg_x = gb.Neg(input_x);
    auto exp_neg_x = gb.Exp(neg_x);
    auto const_one = gb.Const(1.0, input_x->type);
    auto exp_neg_x_add_one = gb.Add(exp_neg_x, const_one);
    auto result = gb.Div(input_dy, exp_neg_x_add_one);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("SoftplusGrad", SoftplusGrad);
}  // namespace mindspore::graphkernel::expanders
