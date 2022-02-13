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

#include "common/graph_kernel/expanders/expander_factory.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class Softplus : public OpDesc {
 public:
  Softplus() {}
  ~Softplus() = default;

 protected:
  bool CheckInputs() override {
    const auto &input_x = inputs_info_[0];
    if (input_x.type != kNumberTypeFloat32 && input_x.type != kNumberTypeFloat16) {
      MS_LOG(INFO) << "In Softplus, input_x's dtype must be float16 or float32";
      return false;
    }
    return true;
  }

  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    const auto &input_x = inputs[0];
    auto exp_x = gb.Emit("Exp", {input_x});
    tensor::TensorPtr data = std::make_shared<tensor::Tensor>(static_cast<double>(1.0), TypeIdToType(input_x->type));
    auto const_one = gb.Value(data);
    auto exp_x_add_one = gb.Emit("Add", {exp_x, const_one});
    auto result = gb.Emit("Log", {exp_x_add_one});
    return {result};
  }
};
OP_EXPANDER_REGISTER("Softplus", Softplus);
}  // namespace mindspore::graphkernel::expanders
