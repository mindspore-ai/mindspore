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

namespace mindspore::graphkernel::expanders {
class Concat : public OpDesc {
 public:
  Concat() {
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Concat() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    auto axis = GetValue<int64_t>(attrs_["axis"]);
    auto result = gb.Concat(inputs, axis);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Concat", Concat);
}  // namespace mindspore::graphkernel::expanders
