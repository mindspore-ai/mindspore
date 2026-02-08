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

#include "backend/ms_backend/graph_fusion/expanders/utils.h"
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"
#include "backend/ms_backend/graph_fusion/model/graph_builder.h"
#include "ir/anf.h"
#include "utils/convert_utils_base.h"

namespace mindspore::graphkernel::expanders {
class ReduceMean : public OpDesc {
 public:
  ReduceMean() {
    std::initializer_list<std::string> attrs{"axis", "keep_dims"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~ReduceMean() = default;

 protected:
  bool CheckInputs() override {
    const auto &x = inputs_info_[0];
    auto x_shape = x.shape;
    if (x_shape.empty() || IsDynamicRank(x_shape)) {
      MS_LOG(INFO) << "Skip empty shape or dynamic rank, shape is: " << x_shape;
      return false;
    }
    axis_ = GetAxisList(attrs_["axis"]);
    auto rank = SizeToLong(x_shape.size());
    (void)std::for_each(axis_.begin(), axis_.end(), [rank](auto &a) { a = a < 0 ? a + rank : a; });
    if (axis_.empty()) {
      for (int64_t i = 0; i < rank; ++i) {
        axis_.push_back(i);
      }
    }
    for (const auto &a : axis_) {
      if (x_shape.at(LongToSize(a)) < 0) {
        MS_LOG(INFO) << "Input shape " << x_shape << " at reduce axis [" << a << "] is dynamic";
        return false;
      }
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    auto x = inputs[0];
    auto dtype = x->type;
    if (dtype != kNumberTypeFloat32) {
      x = gb.Cast(x, kNumberTypeFloat32);
    }
    int64_t sz = 1;
    for (size_t i = 0; i < x->shape.size(); ++i) {
      if (std::find(axis_.begin(), axis_.end(), SizeToLong(i)) != axis_.end()) {
        sz *= x->shape[i];
      }
    }
    auto sum_x = gb.ReduceSum(x, axis_, GetValue<bool>(attrs_["keep_dims"]));
    auto result = gb.Div(sum_x, gb.Tensor(sz, x->type));
    if (dtype != kNumberTypeFloat32) {
      result = gb.Cast(result, dtype);
    }
    return {result};
  }

 private:
  std::vector<int64_t> axis_;
};
EXPANDER_OP_DESC_REGISTER("ReduceMean", ReduceMean);
}  // namespace mindspore::graphkernel::expanders
