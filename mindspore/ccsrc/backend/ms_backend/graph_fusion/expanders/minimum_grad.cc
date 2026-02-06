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
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
constexpr int64_t FAIL = -1;

class MinimumGrad : public OpDesc {
 public:
  MinimumGrad() { (void)validators_.emplace_back(std::make_unique<CheckAllFormatsSame>()); }
  ~MinimumGrad() = default;

 private:
  // Compute reduce axis for final output_shape, {FAIL} for fail
  static ShapeVector GetReduceAxis(const ShapeVector &original_shape, const ShapeVector &brodcast_shape) {
    if (original_shape.size() > brodcast_shape.size()) {
      MS_LOG(INFO) << "For 'MinimumGrad', the length of original_shape should be less than brodcast_shape, but got "
                   << original_shape << " and " << brodcast_shape;
      return {FAIL};
    }
    auto tmp_shape = ShapeVector(brodcast_shape.size() - original_shape.size(), 1);
    (void)tmp_shape.insert(tmp_shape.end(), original_shape.begin(), original_shape.end());

    ShapeVector reduce_axis;
    for (size_t i = 0; i < tmp_shape.size(); ++i) {
      if (tmp_shape[i] != brodcast_shape[i]) {
        if (tmp_shape[i] == 1) {
          reduce_axis.push_back(SizeToLong(i));
        } else {
          MS_LOG(INFO) << "For MinimumGrad, original_shape " << original_shape << " and brodcast_shape "
                       << brodcast_shape << " can't get reduce axis";
          return {FAIL};
        }
      }
    }
    return reduce_axis;
  }

 protected:
  bool CheckInputs() override {
    auto grad_x = true;
    if (attrs_.find("grad_x") != attrs_.end()) {
      grad_x = GetValue<bool>(attrs_["grad_x"]);
    }
    auto grad_y = true;
    if (attrs_.find("grad_y") != attrs_.end()) {
      grad_y = GetValue<bool>(attrs_["grad_y"]);
    }
    if (!grad_x && !grad_y) {
      MS_LOG(INFO) << "For 'MinimumGrad', value of attr 'grad_x' and 'grad_y' should be true, but got " << grad_x
                   << " and " << grad_y;
      return false;
    }
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &input_x = inputs[0];
    const auto &input_y = inputs[1];
    const auto &input_dout = inputs[2];

    auto le_result = gb.LessEqual(input_x, input_y);
    le_result = gb.Cast(le_result, input_x->type);

    auto dx = gb.Mul(le_result, input_dout);
    auto dy = gb.Sub(input_dout, dx);

    auto reduce_axis_x = GetReduceAxis(input_x->shape, dx->shape);
    auto dx_out = dx;
    if (!reduce_axis_x.empty()) {
      if (reduce_axis_x[0] == FAIL) {
        // GetReduceAxis failed. return empty list to make CheckOutputs stop expanding
        return {};
      }
      auto dx_reduce = gb.ReduceSum(dx, reduce_axis_x, false);
      if (dx_reduce->shape != input_x->shape) {
        dx_out = gb.Reshape(dx_reduce, input_x->shape);
      } else {
        dx_out = dx_reduce;
      }
    }

    auto reduce_axis_y = GetReduceAxis(input_y->shape, dy->shape);
    auto dy_out = dy;
    if (!reduce_axis_y.empty()) {
      if (reduce_axis_y[0] == FAIL) {
        // GetReduceAxis failed. return empty list to make CheckOutputs stop expanding
        return {};
      }
      auto dy_reduce = gb.ReduceSum(dy, reduce_axis_y, false);
      if (dy_reduce->shape != input_y->shape) {
        dy_out = gb.Reshape(dy_reduce, input_y->shape);
      } else {
        dy_out = dy_reduce;
      }
    }

    return {dx_out, dy_out};
  }
};
EXPANDER_OP_DESC_REGISTER("MinimumGrad", MinimumGrad);
}  // namespace mindspore::graphkernel::expanders
