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
#include "backend/common/graph_kernel/expanders/utils.h"
#include "kernel/common_utils.h"

namespace mindspore::graphkernel::expanders {
class Softmax : public OpDesc {
 public:
  Softmax() {
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Softmax() = default;

 protected:
  bool CheckInputs() override {
    const auto &var = inputs_info_[0];
    if (Processor() == kernel::kProcessorAiCore &&
        (var.format != kOpFormat_FRAC_NZ || var.format != kOpFormat_DEFAULT)) {
      MS_LOG(INFO) << "Only support default format and FRAC_NZ format on Ascend";
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input = inputs[0];
    auto ori_shape = input->shape;
    if (input->format == kOpFormat_FRAC_NZ) {
      ori_shape = InferShapeFromFractalnz(ori_shape);
    }
    std::vector<int64_t> axis = GetAxisList(attrs_["axis"]);
    for (size_t i = 0; i < axis.size(); i++) {
      if (axis[i] < 0) {
        axis[i] += SizeToLong(input->shape.size());
      }
    }
    auto ori_reduced_shape = GetReducedOriShape(ori_shape, axis);
    auto reduce_axis = axis;
    if (input->format == kOpFormat_FRAC_NZ) {
      reduce_axis = ToFracZAxis(ori_shape, axis);
    }
    NodePtr max_x;
    NodePtr real_input = input;
    if (input->type != TypeId::kNumberTypeFloat16 && Processor() == kernel::kProcessorAiCore) {
      auto input_f16 = gb.Cast(input, TypeId::kNumberTypeFloat16);
      auto max_x_f16 = gb.ReduceMax(input_f16, reduce_axis, true);
      max_x = gb.Cast(max_x_f16, input->type);
    } else {
      max_x = gb.ReduceMax(input, reduce_axis, true);
    }
    if (input->type == TypeId::kNumberTypeFloat16 && Processor() == kernel::kProcessorAiCore) {
      max_x = gb.Cast(max_x, TypeId::kNumberTypeFloat32);
      real_input = gb.Cast(input, TypeId::kNumberTypeFloat32);
    }
    if (input->format == kOpFormat_FRAC_NZ) {
      max_x = gb.Reshape(max_x, ori_reduced_shape);
    }
    auto sub_exp = gb.Exp(gb.Sub(real_input, max_x));
    auto sum_v = gb.ReduceSum(sub_exp, reduce_axis, true);
    if (input->format == kOpFormat_FRAC_NZ) {
      sum_v = gb.Reshape(sum_v, ori_reduced_shape);
    }
    auto result = gb.Div(sub_exp, sum_v);
    if (input->type == TypeId::kNumberTypeFloat16 && Processor() == kernel::kProcessorAiCore) {
      result = gb.Cast(result, input->type);
    }
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Softmax", Softmax);
}  // namespace mindspore::graphkernel::expanders
