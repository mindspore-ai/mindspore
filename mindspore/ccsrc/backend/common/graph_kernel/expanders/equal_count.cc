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
class EqualCount : public OpDesc {
 public:
  EqualCount() {}
  ~EqualCount() = default;

 protected:
  bool CheckInputs() override {
    auto it = std::find_if(std::begin(inputs_info_), std::end(inputs_info_), [](const inner::NodeBase &input) {
      return input.type != kNumberTypeFloat32 && input.type != kNumberTypeFloat16 && input.type != kNumberTypeInt32;
    });
    if (it != std::end(inputs_info_)) {
      MS_LOG(INFO) << "In EqualCount, input's dtype must be float16 or float32 or int32, But input's type is "
                   << it->type;
      return false;
    }
    const auto &input_x = inputs_info_[0];
    const auto &input_y = inputs_info_[1];
    if (input_x.type != input_y.type) {
      MS_LOG(INFO) << "In EqualCount, the inputs data type should be same, But input_x's type is " << input_x.type
                   << " input_y's type is " << input_y.type;
      return false;
    }
    if (input_x.shape != input_y.shape) {
      MS_LOG(INFO) << "In EqualCount, the inputs data shape should be same, But input_x's shape is " << input_x.shape
                   << " input_y's shape is " << input_y.shape;
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_y = inputs[1];
    auto dtype = input_x->type;
    auto eql_val = gb.Equal(input_x, input_y);
    auto cast_val = gb.Cast(eql_val, kNumberTypeFloat32);
    auto shape_size = input_x->shape.size();
    std::vector<int64_t> axis(shape_size);
    for (size_t i = 0; i < shape_size; ++i) {
      axis[i] = SizeToLong(i);
    }
    auto result = gb.ReduceSum(cast_val, axis, false);
    result = gb.Reshape(result, {1});
    if (result->type != dtype) {
      result = gb.Cast(result, dtype);
    }
    return {result};
  }
};
EXPANDER_OP_DESC_REGISTER("EqualCount", EqualCount);
}  // namespace mindspore::graphkernel::expanders
