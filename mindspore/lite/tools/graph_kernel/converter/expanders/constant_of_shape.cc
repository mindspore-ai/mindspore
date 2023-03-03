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
class ConstantOfShape : public OpDesc {
 public:
  ConstantOfShape() {
    std::initializer_list<std::string> attrs{"value", "data_type", "shape"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~ConstantOfShape() = default;

 protected:
  NodePtrList Expand(const NodePtrList &) override {
    auto result = gb.Emit("ConstantOfShape", {},
                          {{"value", attrs_["value"]}, {"data_type", attrs_["data_type"]}, {"shape", attrs_["shape"]}});
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("ConstantOfShape", ConstantOfShape);
}  // namespace mindspore::graphkernel::expanders
