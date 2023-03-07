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

#include <memory>
#include <vector>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class AssignAdd : public OpDesc {
 public:
  AssignAdd() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~AssignAdd() = default;

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_param = inputs[0];
    const auto &input_x = inputs[1];
    auto next_para = gb.Add(input_param, input_x);
    auto result = gb.Assign(input_param, next_para);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("AssignAdd", AssignAdd);
}  // namespace mindspore::graphkernel::expanders
