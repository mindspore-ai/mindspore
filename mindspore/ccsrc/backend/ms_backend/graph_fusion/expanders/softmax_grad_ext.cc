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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "backend/ms_backend/graph_fusion/expanders/op_desc_registry.h"

namespace mindspore::graphkernel::expanders {
class SoftmaxGradExt : public OpDesc {
 public:
  SoftmaxGradExt() {
    auto support_format = std::make_unique<SupportFormat>();
    support_format->AddFormat({kOpFormat_FRAC_Z, kOpFormat_FRAC_NZ, kOpFormat_DEFAULT});
    support_format->AddFormat({kOpFormat_DEFAULT, kOpFormat_DEFAULT, kOpFormat_DEFAULT});
    (void)validators_.emplace_back(std::move(support_format));
    std::initializer_list<std::string> attrs{"axis"};
    (void)validators_.emplace_back(std::make_unique<CheckAttr>(attrs));
  }
  ~SoftmaxGradExt() = default;

 private:
  static std::tuple<ShapeVector, ShapeVector> GetReduceAxisShape(const ShapeVector &shape,
                                                                 const std::string &data_format, ShapeVector &&axis) {
    auto ori_shape = shape;
    if (data_format == kOpFormat_FRAC_NZ) {
      ori_shape = InferShapeFromFractalnz(shape);
    }
    (void)std::for_each(axis.begin(), axis.end(), [&ori_shape](auto &ele) {
      if (ele < 0) {
        ele += static_cast<int64_t>(ori_shape.size());
      }
    });

    auto reduced_ori_shape = GetReducedOriShape(ori_shape, axis);
    auto reduce_axis = axis;
    if (data_format == kOpFormat_FRAC_NZ) {
      reduce_axis = ToFracZAxis(ori_shape, axis);
    }
    return {reduce_axis, reduced_ori_shape};
  }

 protected:
  bool CheckInputs() override {
    const auto &x = inputs_info_[0];
    if (IsDynamicRank(x.shape)) {
      MS_LOG(DEBUG) << "Skip dynamic rank case";
      return false;
    }
    auto axis = GetAxisList(attrs_["axis"]);
    auto [reduce_axis, ori_reduced_shape] = GetReduceAxisShape(x.shape, x.format, std::move(axis));
    if (x.format == kOpFormat_FRAC_NZ &&
        std::count_if(ori_reduced_shape.begin(), ori_reduced_shape.end(), [](int64_t sh) { return sh < 0; }) > 1) {
      MS_LOG(DEBUG) << "Skip dynamic shape case";
      return false;
    }
    reduce_axis_ = reduce_axis;
    ori_reduced_shape_ = ori_reduced_shape;
    return true;
  }

  NodePtrList Expand(const NodePtrList &inputs) override {
    const auto &x = inputs[0];
    const auto &y = inputs[1];
    const auto &z = inputs[2];
    auto data_mul = gb.Mul(x, y);
    auto data_sum = gb.Emit("ReduceSum", {data_mul, gb.Tensor(reduce_axis_)},
                            {{"keep_dims", MakeValue(true)}, {"reduce_output_fuse", MakeValue(true)}});
    if (x->format == kOpFormat_FRAC_NZ) {
      data_sum = gb.Reshape(data_sum, ori_reduced_shape_);
    }
    auto data_sub = gb.Sub(x, data_sum);
    auto data_mul2 = gb.Mul(data_sub, y);
    auto result = gb.Mul(data_mul2, z);
    return {result};
  }

  ShapeVector reduce_axis_;
  ShapeVector ori_reduced_shape_;
};
EXPANDER_OP_DESC_REGISTER("SoftmaxGradExt", SoftmaxGradExt);
}  // namespace mindspore::graphkernel::expanders
