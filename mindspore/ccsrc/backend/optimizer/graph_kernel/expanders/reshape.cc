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

#include "backend/optimizer/graph_kernel/expanders/expander_factory.h"

namespace mindspore {
namespace opt {
namespace expanders {
class ExpandDims : public OpExpander {
 public:
  ExpandDims() {
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~ExpandDims() = default;

  static ShapeVector InferShape(const ShapeVector &shape, const std::vector<int64_t> &axis) {
    ShapeVector new_shape = shape;
    for (auto x : axis) {
      int64_t rank = static_cast<int64_t>(new_shape.size());
      if (x > rank || x < -rank - 1) {
        MS_LOG(EXCEPTION) << "ExpandDims axis " << x << " is out of range of size " << new_shape.size();
      }
      if (x >= 0) {
        (void)new_shape.insert(new_shape.begin() + x, 1LL);
      } else {
        (void)new_shape.insert(new_shape.begin() + (x + rank + 1), 1LL);
      }
    }
    return new_shape;
  }

 protected:
  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    const auto &input_x = inputs[0];
    auto shape = MakeValue(ExpandDims::InferShape(input_x->shape, GetAxisList(this->attrs_["axis"])));
    auto result = gb.Emit("Reshape", {input_x}, {{"shape", shape}});
    return {result};
  }
};
OP_EXPANDER_REGISTER("ExpandDims", ExpandDims);

ShapeVector ExpandDimsInferShape(const ShapeVector &shape, const std::vector<int64_t> &axis) {
  return ExpandDims::InferShape(shape, axis);
}
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
