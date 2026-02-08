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
class LogSoftmax : public OpDesc {
 public:
  LogSoftmax() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~LogSoftmax() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    auto axis = GetAxisList(attrs_["axis"]);

    auto x_f16 = input_x;
    NodePtr max_x;
    auto ori_dtype = input_x->type;
    if (ori_dtype != TypeId::kNumberTypeFloat16 && processor_ == "aicore") {
      x_f16 = gb.Cast(x_f16, TypeId::kNumberTypeFloat16);
      auto max_x_f16 = gb.ReduceMax(x_f16, axis, true);
      max_x = gb.Cast(max_x_f16, ori_dtype);
    } else {
      max_x = gb.ReduceMax(x_f16, axis, true);
    }

    auto data_sub = gb.Sub(input_x, max_x);
    auto data_exp = gb.Exp(data_sub);
    auto data_exp_sum = gb.ReduceSum(data_exp, axis, true);
    auto log_exp_sum = gb.Log(data_exp_sum);
    auto result = gb.Sub(data_sub, log_exp_sum);

    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("LogSoftmax", LogSoftmax);
}  // namespace mindspore::graphkernel::expanders
