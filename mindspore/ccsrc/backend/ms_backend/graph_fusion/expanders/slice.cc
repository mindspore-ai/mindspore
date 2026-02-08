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
class Slice : public OpDesc {
 public:
  Slice() {
    std::initializer_list<std::string> attrs{"begin", "size"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Slice() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &begin = GetAxisList(attrs_["begin"]);
    const auto &size = GetAxisList(attrs_["size"]);
    ShapeVector end;
    ShapeVector strides;
    for (size_t i = 0; i < begin.size(); ++i) {
      strides.push_back(1);
      end.push_back(begin[i] + size[i]);
    }
    auto result = gb.StridedSlice(input_x, begin, end, strides);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Slice", Slice);
}  // namespace mindspore::graphkernel::expanders
