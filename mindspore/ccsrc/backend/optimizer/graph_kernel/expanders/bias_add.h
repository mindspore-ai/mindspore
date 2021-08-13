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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_BIAS_ADD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_BIAS_ADD_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "backend/optimizer/graph_kernel/expanders/utils.h"

namespace mindspore {
namespace opt {
namespace expanders {
class BiasAdd : public OpExpander {
 public:
  BiasAdd() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_NCHW, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_NHWC, kOpFormat_DEFAULT});
    validators_.emplace_back(std::move(support_format));
    validators_.emplace_back(new CheckAttr({"format"}));
  }
  ~BiasAdd() = default;
  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    auto input_x = inputs[0];
    auto input_y = inputs[1];
    if (input_x->format == kOpFormat_NCHW) {
      input_y = gb.Emit("Reshape", {input_y}, {{"shape", MakeValue(ExpandDims::InferShape(input_y->shape, {1, 2}))}});
    } else if (input_x->format == kOpFormat_DEFAULT) {
      auto data_format = GetValue<std::string>(attrs_["format"]);
      size_t channel_idx = (data_format == kOpFormat_NHWC) ? input_x->shape.size() - 1 : 1;
      std::vector<int64_t> axis(input_x->shape.size() - channel_idx - 1, -1);
      if (!axis.empty()) {
        input_y = gb.Emit("Reshape", {input_y}, {{"shape", MakeValue(ExpandDims::InferShape(input_y->shape, axis))}});
      }
    }
    return {gb.Emit("Add", {input_x, input_y})};
  }
};
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_BIAS_ADD_H_
