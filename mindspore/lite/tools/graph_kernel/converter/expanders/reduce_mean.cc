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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class ReduceMean : public OpDesc {
 public:
  ReduceMean() {
    std::initializer_list<std::string> attrs{"axis", "keep_dims"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~ReduceMean() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    auto rank = SizeToLong(x->shape.size());
    auto axis = GetAxisList(attrs_["axis"]);
    (void)std::for_each(axis.begin(), axis.end(), [rank](auto &a) { a = a < 0 ? a + rank : a; });
    if (axis.empty()) {
      for (int64_t i = 0; i < rank; ++i) {
        axis.push_back(i);
      }
    }
    int64_t sz = 1;
    for (size_t i = 0; i < x->shape.size(); ++i) {
      if (std::find(axis.begin(), axis.end(), SizeToLong(i)) != axis.end()) {
        sz *= SizeToLong(x->shape[i]);
      }
    }
    auto sum_x = gb.ReduceSum(x, axis, GetValue<bool>(attrs_["keep_dims"]));
    auto result = gb.Div(sum_x, gb.Const(sz, x->type));
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("ReduceMean", ReduceMean);
}  // namespace mindspore::graphkernel::expanders
