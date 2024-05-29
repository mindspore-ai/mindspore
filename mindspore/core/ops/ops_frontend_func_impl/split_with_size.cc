/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <map>
#include <memory>
#include "ops/ops_frontend_func_impl.h"
#include "ops/op_utils.h"
#include "utils/log_adapter.h"
#include "abstract/abstract_value.h"

namespace mindspore {
namespace ops {

class SplitWithSizeFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    auto input_abs = input_args[kIndex0];
    auto input_shape_ptr = input_abs->GetShape();
    auto input_shape = input_shape_ptr->GetShapeVector();
    auto axis_opt = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
    auto split_size_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
    AbstractBasePtrList output_list;
    if (!split_size_opt.has_value()) {
      auto dynamic_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      (void)output_list.push_back(abstract::MakeAbstractTensor(dynamic_shape, input_abs->GetType()));
      auto abs_tuple = std::make_shared<abstract::AbstractTuple>(output_list);
      abs_tuple->CheckAndConvertToDynamicLenSequence();
      return abs_tuple;
    }
    auto split_size = split_size_opt.value();
    if (IsDynamicRank(input_shape)) {
      auto dynamic_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
      for (size_t i = 0; i < split_size.size(); ++i) {
        (void)output_list.push_back(abstract::MakeAbstractTensor(dynamic_shape, input_abs->GetType()));
      }
    } else if (!axis_opt.has_value()) {
      auto dynamic_shape =
        std::make_shared<abstract::Shape>(std::vector<int64_t>(input_shape.size(), abstract::Shape::kShapeDimAny));
      for (size_t i = 0; i < split_size.size(); ++i) {
        (void)output_list.push_back(abstract::MakeAbstractTensor(dynamic_shape, input_abs->GetType()));
      }
    } else {
      auto output_shape = input_shape;
      auto rank = SizeToLong(input_shape.size());
      auto axis = axis_opt.value();
      if (axis < 0) {
        axis += rank;
      }
      size_t pos = LongToSize(axis);
      int64_t sum_split_size = std::accumulate(split_size.ToVector().begin(), split_size.ToVector().end(), 0);
      if (sum_split_size != output_shape[pos]) {
        MS_EXCEPTION(ValueError) << "split_size's length must be equal with dimIndex";
      }
      for (size_t i = 0; i < split_size.size(); ++i) {
        if (split_size.IsValueUnknown(i)) {
          output_shape[pos] = abstract::Shape::kShapeDimAny;
        } else {
          output_shape[pos] = split_size[i];
        }
        auto shape = std::make_shared<abstract::Shape>(output_shape);
        auto abs = abstract::MakeAbstractTensor(shape, input_abs->GetType());
        (void)output_list.push_back(abs);
      }
    }
    auto abs = std::make_shared<abstract::AbstractTuple>(output_list);
    return abs;
  }
};
REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("SplitWithSize", SplitWithSizeFrontendFuncImpl);
}  // namespace ops
}  // namespace mindspore
