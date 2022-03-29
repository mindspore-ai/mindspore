/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "common/graph_kernel/expanders/expander_factory.h"
#include "common/graph_kernel/expanders/utils.h"

namespace mindspore::graphkernel::expanders {
class BiasAdd : public OpDesc {
 public:
  BiasAdd() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_NCHW, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_NHWC, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
    auto attrs = std::initializer_list<std::string>{"format"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~BiasAdd() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    auto input_x = inputs[0];
    auto input_y = inputs[1];
    if (input_x->format == kOpFormat_NCHW) {
      auto target_shape = ExpandDimsInferShape(input_y->shape, {1, 2});
      input_y = gb.Reshape(input_y, target_shape);
    } else if (input_x->format == kOpFormat_DEFAULT) {
      auto data_format = GetValue<std::string>(attrs_["format"]);
      size_t channel_idx = (data_format == kOpFormat_NHWC) ? input_x->shape.size() - 1 : 1;
      std::vector<int64_t> axis((input_x->shape.size() - channel_idx) - 1, -1);
      if (!axis.empty()) {
        auto target_shape = ExpandDimsInferShape(input_y->shape, axis);
        input_y = gb.Reshape(input_y, target_shape);
      }
    }
    auto result = gb.Add(input_x, input_y);
    return {result};
  }
};
OP_EXPANDER_REGISTER("BiasAdd", BiasAdd);
}  // namespace mindspore::graphkernel::expanders
