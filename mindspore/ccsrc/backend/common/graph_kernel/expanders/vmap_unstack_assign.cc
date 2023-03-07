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
#include <vector>

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
constexpr size_t kInputLowerLimit = 4;
constexpr size_t kStackedParamIndex = 0;
constexpr size_t kNumber1 = 1;
constexpr size_t kNumber2 = 2;

class VmapUnstackAssign : public OpDesc {
 public:
  VmapUnstackAssign() {}
  ~VmapUnstackAssign() = default;

 protected:
  bool CheckInputs() override {
    inputs_size_ = inputs_info_.size();
    if (inputs_size_ <= kInputLowerLimit) {
      MS_LOG(INFO) << "In VmapUnstackAssign, inputs size must be greater than 4, but got " << inputs_size_ << ".";
      return false;
    }

    // The input format is: stacked parameter, param1, param2, ...(a batch of parameters), UMonad.
    params_size_ = inputs_size_ - kNumber2;
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    auto input_param = gb.Emit("Depend", {inputs[kStackedParamIndex], inputs.back()});
    auto unstacked_para =
      gb.Emit("Unstack", {input_param},
              {{"axis", MakeValue(static_cast<int64_t>(0))}, {"num", MakeValue(static_cast<int64_t>(params_size_))}});

    NodePtrList res;
    for (size_t i = 0; i < params_size_; i++) {
      auto getitem = gb.TupleGetItem(unstacked_para, SizeToLong(i));
      auto result = gb.Assign(inputs[i + kNumber1], getitem);
      res.push_back(result);
    }
    auto res_tuple = gb.Emit("MakeTuple", res);
    auto result = gb.Emit("Depend", {gb.Reshape(gb.Const(kNumber1, kNumberTypeInt32), {kNumber1}), res_tuple});
    return {result};
  }

 private:
  size_t inputs_size_{0};
  size_t params_size_{0};
};
EXPANDER_OP_DESC_REGISTER("VmapUnstackAssign", VmapUnstackAssign);
}  // namespace mindspore::graphkernel::expanders
