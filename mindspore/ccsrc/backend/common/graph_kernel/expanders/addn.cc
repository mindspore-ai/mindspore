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
#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class AddN : public OpDesc {
 public:
  AddN() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~AddN() = default;

  static NodePtr Exec(const inner::GraphBuilder &gb, const NodePtrList &inputs) {
    auto result = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
      result = gb.Add(result, inputs[i]);
    }
    return result;
  }

 protected:
  bool CheckInputs() override {
    constexpr size_t min_inputs = 2;
    if (inputs_info_.size() < min_inputs) {
      MS_LOG(INFO) << "For 'AddN', the inputs num should be greater than 1, but got " << inputs_info_.size();
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override { return {Exec(gb, inputs)}; }
};
EXPANDER_OP_DESC_REGISTER("AddN", AddN);
}  // namespace mindspore::graphkernel::expanders
