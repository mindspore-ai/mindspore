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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class ArgMaxMinWithValue : public OpDesc {
 public:
  ArgMaxMinWithValue() {
    std::initializer_list<std::string> attrs{"axis", "keep_dims"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~ArgMaxMinWithValue() = default;

 protected:
  bool CheckInputs() override {
    const auto &var = inputs_info_[0];
    if (var.type != kNumberTypeFloat32 && var.type != kNumberTypeFloat16) {
      MS_LOG(INFO) << "In ArgMax(Min)WithValue, var's dtype must be float16 or float32";
      return false;
    }
    return true;
  }

  // ArgMax(Min)WithValue will be expanded to [Argmax(min)(+Reshape), ReduceMax(Min)]
  // Currently only expand it when output[1] has users and output[0] has no users
  // Thus, Argmax(min)(+Reshape) will be eliminated in later pass because of no users
  // and ArgMax(Min)WithValue will be finally converted to ReduceMax(Min)
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    std::vector<int64_t> shape = outputs_info_[0].shape;
    auto axis = GetValue<int64_t>(attrs_["axis"]);
    auto keep_dims = GetValue<bool>(attrs_["keep_dims"]);
    std::string argmaxmin = this->Name() == "ArgMaxWithValue" ? "Argmax" : "Argmin";
    auto index =
      gb.Emit(argmaxmin, {input_x},
              {{"axis", MakeValue(std::vector<int64_t>({axis}))}, {"output_type", TypeIdToType(kNumberTypeInt32)}});
    if (keep_dims) {
      index = gb.Reshape(index, shape);
    }

    auto value = this->Name() == "ArgMaxWithValue" ? gb.ReduceMax(input_x, std::vector<int64_t>({axis}), keep_dims)
                                                   : gb.ReduceMin(input_x, std::vector<int64_t>({axis}), keep_dims);

    auto result = {index, value};
    return result;
  }
};
EXPANDER_OP_DESC_REGISTER("ArgMaxWithValue", ArgMaxMinWithValue);
EXPANDER_OP_DESC_REGISTER("ArgMinWithValue", ArgMaxMinWithValue);
}  // namespace mindspore::graphkernel::expanders
