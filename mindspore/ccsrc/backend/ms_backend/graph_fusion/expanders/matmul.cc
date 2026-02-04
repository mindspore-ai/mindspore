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
#include <string>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class MatMul : public OpDesc {
 public:
  MatMul() {
    std::initializer_list<std::string> attrs{"transpose_a", "transpose_b", "left_format", "right_format"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~MatMul() = default;

 private:
  static void Transpose(ShapeVector *shape) {
    auto shape_len = shape->size();
    // transpose last two axis
    auto tmp = shape[shape_len - kDim1];
    shape[shape_len - kDim1] = shape[shape_len - kDim2];
    shape[shape_len - kDim2] = tmp;
  }

 protected:
  void Init() override {
    constexpr size_t idx_a = 0;
    constexpr size_t idx_b = 1;
    shape_a_ = inputs_info_[idx_a].shape;
    shape_b_ = inputs_info_[idx_b].shape;
    transpose_a_ = GetValue<bool>(attrs_["transpose_a"]);
    transpose_b_ = GetValue<bool>(attrs_["transpose_b"]);
    left_format_ = GetValue<std::string>(attrs_["left_format"]);
    right_format_ = GetValue<std::string>(attrs_["right_format"]);
  }
  bool CheckInputs() override {
    if (processor_ != "aicore" || left_format_ != kOpFormat_DEFAULT || right_format_ != kOpFormat_DEFAULT) {
      MS_LOG(INFO) << "MatMul/BatchMatMul do not need to be replaced by Mul";
      return false;
    }
    if (transpose_a_ && shape_a_.size() < kDim2) {
      MS_LOG(INFO) << "shape of input_0 should be bigger than 2 but got " << shape_a_.size();
      return false;
    }
    if (transpose_b_ && shape_b_.size() < kDim2) {
      MS_LOG(INFO) << "shape of input_1 should be bigger than 2 but got " << shape_b_.size();
      return false;
    }
    auto k_a = transpose_a_ ? shape_a_[shape_a_.size() - 2] : shape_a_[shape_a_.size() - 1];
    auto k_b = transpose_b_ ? shape_b_[shape_b_.size() - 1] : shape_b_[shape_b_.size() - 2];
    if (k_a != 1 || k_b != 1) {
      MS_LOG(INFO) << "MatMul/BatchMatMul can't be expanded when k != 1";
      return false;
    }
    return true;
  }
  NodePtrList Expand(const NodePtrList &inputs) override {
    auto a = inputs[0];
    auto b = inputs[1];

    if (transpose_a_) {
      Transpose(&shape_a_);
      a = gb.Reshape(a, shape_a_);
    }
    if (transpose_b_) {
      Transpose(&shape_b_);
      b = gb.Reshape(b, shape_b_);
    }

    auto result = gb.Mul(a, b);
    if (attrs_.find("dst_type") != attrs_.end()) {
      auto dtype = attrs_["dst_type"]->cast<TypePtr>();
      if (dtype != nullptr) {
        if (dtype != TypeIdToType(result->type)) {
          result = gb.Cast(result, dtype->type_id());
        }
      }
    }
    return {result};
  }

  std::string left_format_ = {};
  std::string right_format_ = {};
  bool transpose_a_ = false;
  bool transpose_b_ = false;
  ShapeVector shape_a_ = {};
  ShapeVector shape_b_ = {};
};
EXPANDER_OP_DESC_REGISTER("MatMul", MatMul);
EXPANDER_OP_DESC_REGISTER("BatchMatMul", MatMul);
}  // namespace mindspore::graphkernel::expanders
