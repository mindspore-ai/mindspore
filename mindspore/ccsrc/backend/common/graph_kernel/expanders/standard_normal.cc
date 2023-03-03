/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

namespace mindspore::graphkernel::expanders {
class StandardNormal : public OpDesc {
 public:
  StandardNormal() {
    std::initializer_list<std::string> attrs{"seed", "seed2"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~StandardNormal() {}

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    auto shape = MakeValue(outputs_info_[0].shape);
    auto result =
      gb.Emit("StandardNormal", {input_x}, {{"shape", shape}, {"seed", attrs_["seed"]}, {"seed2", attrs_["seed2"]}});
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("StandardNormal", StandardNormal);
}  // namespace mindspore::graphkernel::expanders
