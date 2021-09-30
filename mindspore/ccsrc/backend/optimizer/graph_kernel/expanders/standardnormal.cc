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

#include "backend/optimizer/graph_kernel/expanders/expander_factory.h"

namespace mindspore {
namespace opt {
namespace expanders {
class StandardNormal : public OpExpander {
 public:
  StandardNormal() {
    std::initializer_list<std::string> attrs{"seed", "seed2"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~StandardNormal() {}

 protected:
  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    const auto &input_x = inputs[0];
    auto shape = MakeValue(outputs_info_[0].shape);
    auto result =
      gb.Emit("StandardNormal", {input_x}, {{"shape", shape}, {"seed", attrs_["seed"]}, {"seed2", attrs_["seed2"]}});
    return {result};
  }
};
OP_EXPANDER_REGISTER("StandardNormal", StandardNormal);
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
