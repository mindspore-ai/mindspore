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

#include "backend/common/graph_kernel/expanders/op_desc_registry.h"
#include "tools/graph_kernel/converter/expanders/activation.h"
#include "mindapi/base/types.h"
#include "ir/dtype.h"

namespace mindspore::graphkernel::expanders {
constexpr size_t kInputIdx = 0;
constexpr size_t kScaleIdx = 1;
constexpr size_t kOffsetIdx = 2;
class ScaleFusion : public OpDesc {
 public:
  ScaleFusion() {
    (void)validators_.emplace_back(std::make_unique<CheckActivationType>(ActivationType::NO_ACTIVATION));
  }
  ~ScaleFusion() = default;

 protected:
  bool CheckInputs() override {
    auto axis = GetAxisList(attrs_["axis"])[0];
    size_t input_shape_size = inputs_info_[kInputIdx].shape.size();
    size_t scale_shape_size = inputs_info_[kScaleIdx].shape.size();
    axis = axis < 0 ? axis + SizeToLong(input_shape_size) : axis;
    return (LongToSize(axis) + scale_shape_size) == input_shape_size;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[kInputIdx];
    const auto &input_scale = inputs[kScaleIdx];
    const auto &input_offset = inputs[kOffsetIdx];
    auto mul = gb.Mul(input_x, input_scale);
    auto mul_add = gb.Add(mul, input_offset);
    return {mul_add};
  }
};
EXPANDER_OP_DESC_REGISTER("ScaleFusion", ScaleFusion);
}  // namespace mindspore::graphkernel::expanders
