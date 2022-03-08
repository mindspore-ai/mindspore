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
#include <set>
#include <string>

#include "common/graph_kernel/expanders/expander_factory.h"
#include "common/graph_kernel/lite_adapter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
NodePtr GetActivationExpander(const inner::LiteGraph::GraphBuilder &gb, const NodePtrList &inputs,
                              int64_t activation_type) {
  switch (activation_type) {
    case ActivationType::RELU:
      return ReluExpand(gb, inputs);
    case ActivationType::SIGMOID:
      return SigmoidExpand(gb, inputs);
    default:
      return inputs[0];
  }
}

class Activation : public OpDesc {
 public:
  Activation() {
    std::initializer_list<std::string> attrs{"activation_type"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
    std::set<int64_t> activation_types = {ActivationType::NO_ACTIVATION, ActivationType::RELU, ActivationType::SIGMOID};
    (void)validators_.emplace_back(std::make_unique<CheckActivationType>(activation_types));
  }
  ~Activation() = default;

 protected:
  NodePtrList Expand() override {
    auto activation_type = GetValue<int64_t>(attrs_["activation_type"]);
    return {GetActivationExpander(gb, gb.Get()->inputs(), activation_type)};
  }
};
OP_EXPANDER_REGISTER("Activation", Activation);
}  // namespace mindspore::graphkernel::expanders
