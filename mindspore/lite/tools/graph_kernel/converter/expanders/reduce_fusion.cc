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
#include <numeric>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class CheckReduceMode : public Validator {
 public:
  bool Check(const OpDesc &e) override {
    auto iter = e.Attrs().find("mode");
    if (iter == e.Attrs().end()) {
      MS_LOG(INFO) << "The mode is not found in attrs.";
      return false;
    }
    auto mode = GetValue<int64_t>(iter->second);
    if (mode != ReduceMode::Reduce_Sum && mode != ReduceMode::Reduce_Mean) {
      MS_LOG(INFO) << "Reduce mode " << mode << " not supported yet!";
      return false;
    }
    return true;
  }
};

class ReduceFusion : public OpDesc {
 public:
  ReduceFusion() {
    std::initializer_list<std::string> attrs{"keep_dims", "coeff"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
    (void)validators_.emplace_back(std::make_unique<CheckReduceMode>());
  }
  ~ReduceFusion() = default;

 protected:
  bool CheckInputs() override {
    const auto &x = inputs_info_[0];
    auto x_shape = x.shape;
    auto mode = GetValue<int64_t>(attrs_["mode"]);
    if (mode == ReduceMode::Reduce_Mean) {
      if (attrs_.count("axis") == 0) {
        MS_LOG(INFO) << "Axis is dynamic, and the mode of ReduceFusion is Reduce_Mean, in this case we can not expand "
                        "ReduceFusion.";
        return false;
      } else if (x_shape.empty() || IsDynamicRank(x_shape)) {
        MS_LOG(INFO) << "Skip empty shape or dynamic rank, shape is: " << x_shape;
        return false;
      } else {
        auto axis = GetAxisList(attrs_["axis"]);
        bool is_valid = std::all_of(axis.begin(), axis.end(), [&x_shape](int idx) { return x_shape[idx] > 0; });
        if (!is_valid) {
          MS_LOG(INFO) << "Some dimension size needed in reducemean is not available";
          return false;
        }
      }
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    auto keep_dims = GetValue<bool>(attrs_["keep_dims"]);
    auto mode = GetValue<int64_t>(attrs_["mode"]);
    if (mode == ReduceMode::Reduce_Mean) {
      auto axis = GetAxisList(attrs_["axis"]);
      auto sum_res = gb.ReduceSum(input_x, axis, keep_dims);
      auto coeff = gb.Tensor(GetValue<float>(attrs_["coeff"]), input_x->type);
      auto result = gb.Mul(sum_res, coeff);
      int64_t reduce_size = std::accumulate(axis.begin(), axis.end(), 1,
                                            [input_x](int64_t a, int64_t idx) { return a * input_x->shape[idx]; });
      auto reduce_size_value = gb.Tensor(reduce_size, input_x->type);
      auto mean_res = gb.Div(result, reduce_size_value);
      return {mean_res};
    } else {
      NodePtr sum_res = nullptr;
      if (attrs_.count("axis") == 0) {
        auto &axis = inputs[1];
        sum_res = gb.Emit("ReduceSum", {input_x, axis}, {{"keep_dims", MakeValue(keep_dims)}});
      } else {
        auto axis = GetAxisList(attrs_["axis"]);
        sum_res = gb.ReduceSum(input_x, axis, keep_dims);
      }
      auto coeff = gb.Tensor(GetValue<float>(attrs_["coeff"]), input_x->type);
      auto result = gb.Mul(sum_res, coeff);
      return {result};
    }
  }
};
EXPANDER_OP_DESC_REGISTER("ReduceFusion", ReduceFusion);
}  // namespace mindspore::graphkernel::expanders
