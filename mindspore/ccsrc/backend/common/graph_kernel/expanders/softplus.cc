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
#include <limits>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class Softplus : public OpDesc {
 public:
  Softplus() {}
  ~Softplus() = default;

 protected:
  bool CheckInputs() override {
    const auto &input_x = inputs_info_[0];
    if (input_x.type != kNumberTypeFloat32 && input_x.type != kNumberTypeFloat16) {
      MS_LOG(INFO) << "In Softplus, input_x's dtype must be float16 or float32";
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    constexpr double num_two = 2.0;
    auto threshold = log(std::numeric_limits<double>::epsilon()) + num_two;
    if (input_x->type == kNumberTypeFloat16) {
      threshold = log(std::numeric_limits<float>::epsilon()) + num_two;
    }
    // if x > -t  result = x
    // if x < t result = exp(x)
    // else result = log(1 + exp(x))
    auto exp_x = gb.Exp(input_x);
    auto const_threshold = gb.Const(threshold, input_x->type);
    auto const_neg_threshold = gb.Const(-threshold, input_x->type);
    auto const_one = gb.Const(1.0, input_x->type);
    auto exp_x_add_one = gb.Add(exp_x, const_one);
    auto result = gb.Log(exp_x_add_one);
    auto greater_neg_t = gb.Greater(input_x, const_neg_threshold);
    result = gb.Select(greater_neg_t, input_x, result);
    auto less_t = gb.Less(input_x, const_threshold);
    result = gb.Select(less_t, exp_x, result);

    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Softplus", Softplus);
}  // namespace mindspore::graphkernel::expanders
