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

namespace mindspore::graphkernel::expanders {
class ExpandDims : public OpDesc {
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
        MS_LOG(EXCEPTION) << "ExpandDims attr 'axis' value " << x << " is out of range of [" << (-rank - 1) << ", "
                          << rank << "]";
      }
      if (x >= 0) {
        (void)new_shape.insert(new_shape.cbegin() + x, 1LL);
      } else {
        (void)new_shape.insert(new_shape.cbegin() + (x + rank + 1), 1LL);
      }
    }
    return new_shape;
  }

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    auto target_shape = ExpandDims::InferShape(input_x->shape, GetAxisList(this->attrs_["axis"]));
    auto result = gb.Reshape(input_x, target_shape);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("ExpandDims", ExpandDims);
EXPANDER_OP_DESC_REGISTER("Unsqueeze", ExpandDims);

class Squeeze : public OpDesc {
 public:
  Squeeze() {
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~Squeeze() = default;

  static ShapeVector InferShape(const ShapeVector &shape, const std::vector<int64_t> &axis) {
    ShapeVector new_shape;
    if (axis.size() == 0) {
      for (auto s : shape) {
        if (s != 1) {
          (void)new_shape.emplace_back(s);
        }
      }
    } else {
      auto ndim = SizeToLong(shape.size());
      for (int64_t i = 0; i < ndim; i++) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end() &&
            std::find(axis.begin(), axis.end(), i - ndim) == axis.end()) {
          (void)new_shape.emplace_back(shape[LongToSize(i)]);
        }
      }
    }
    return new_shape;
  }

 protected:
  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    auto target_shape = Squeeze::InferShape(input_x->shape, GetAxisList(this->attrs_["axis"]));
    auto result = gb.Reshape(input_x, target_shape);
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("Squeeze", Squeeze);

ShapeVector ExpandDimsInferShape(const ShapeVector &shape, const std::vector<int64_t> &axis) {
  return ExpandDims::InferShape(shape, axis);
}
}  // namespace mindspore::graphkernel::expanders
