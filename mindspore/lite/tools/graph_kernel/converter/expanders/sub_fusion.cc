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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class SubFusion : public OpDesc {
 public:
  SubFusion() { (void)validators_.emplace_back(std::make_unique<CheckActivationType>(ActivationType::NO_ACTIVATION)); }
  ~SubFusion() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_y = inputs[1];
    auto result = gb.Sub(input_x, input_y);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("SubFusion", SubFusion);
}  // namespace mindspore::graphkernel::expanders
