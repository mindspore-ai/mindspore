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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_RESHAPE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_RESHAPE_H_

#include <memory>
#include <vector>

#include "backend/optimizer/graph_kernel/model/node.h"
#include "backend/optimizer/graph_kernel/expanders/utils.h"

namespace mindspore {
namespace opt {
namespace expanders {
class ExpandDims : public OpExpander {
 public:
  ExpandDims() { validators_.emplace_back(new CheckAttr({"axis"})); }
  ~ExpandDims() {}
  NodePtrList Expand() override {
    const auto &inputs = gb.Get()->inputs();
    auto &input_x = inputs[0];
    auto shape = MakeValue(ExpandDims::InferShape(input_x->shape, GetAxisList(this->attrs_["axis"])));
    auto result = gb.Emit("Reshape", {input_x}, {{"shape", shape}});
    return {result};
  }

  static ShapeVector InferShape(const ShapeVector &shape, const std::vector<int64_t> &axis) {
    ShapeVector new_shape = shape;
    for (auto x : axis) {
      int64_t rank = static_cast<int64_t>(new_shape.size());
      if (x > rank || x < -rank - 1) {
        std::ostringstream oss;
        oss << "ExpandDims axis " << x << " is out of range of size " << new_shape.size();
        throw graphkernel::GKException(oss.str());
      }
      if (x >= 0) {
        new_shape.insert(new_shape.begin() + x, 1LL);
      } else {
        new_shape.insert(new_shape.begin() + (x + rank + 1), 1LL);
      }
    }
    return new_shape;
  }
};
}  // namespace expanders
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_RESHAPE_H_
