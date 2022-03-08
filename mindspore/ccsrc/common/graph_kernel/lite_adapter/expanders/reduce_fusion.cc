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

#include "common/graph_kernel/expanders/expander_factory.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
class CheckReduceMode : public Validator {
 public:
  bool Check(const OpDesc &e) override {
    if (e.Attrs().count("mode") == 0) {
      return false;
    }
    auto mode = GetValue<int64_t>(e.Attrs()["mode"]);
    if (mode != ReduceMode::Reduce_Sum) {
      MS_LOG(INFO) << "Reduce mode " << mode << " not supported yet!";
      return false;
    }
    return true;
  }
};

class ReduceFusion : public OpDesc {
 public:
  ReduceFusion() {
    std::initializer_list<std::string> attrs{"axis", "keep_dims"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
    (void)validators_.emplace_back(std::make_unique<CheckReduceMode>());
  }
  ~ReduceFusion() = default;

 protected:
  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    const auto &input_x = inputs[0];
    auto axis = GetAxisList(attrs_["axis"]);
    auto keep_dims = attrs_["keep_dims"];
    auto result = gb.Emit("ReduceSum", {input_x}, {{"axis", MakeValue(axis)}, {"keep_dims", keep_dims}});
    return {result};
  }
};
OP_EXPANDER_REGISTER("ReduceFusion", ReduceFusion);
}  // namespace mindspore::graphkernel::expanders
