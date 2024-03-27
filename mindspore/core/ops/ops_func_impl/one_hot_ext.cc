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

#include <memory>
#include <set>
#include <string>
#include "ops/ops_func_impl/one_hot_ext.h"
#include "ops/op_utils.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr OneHotExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &in_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  if (std::any_of(in_shape.begin(), in_shape.end(), [](int64_t s) { return s == 0; })) {
    MS_LOG(EXCEPTION) << "Shape of input should not contain 0, bug got shape: " << in_shape;
  }

  ShapeVector output_shape = in_shape;
  if (IsDynamicRank(output_shape)) {
    return std::make_shared<abstract::Shape>(output_shape);
  }
  auto depth_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  int64_t depth_value;
  if (!depth_opt.has_value()) {
    depth_value = abstract::Shape::kShapeDimAny;
  } else {
    depth_value = depth_opt.value();
    MS_CHECK_VALUE(depth_value >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg("num_classes value", depth_value,
                                                                                 kGreaterEqual, 0, primitive));
  }

  auto axis_opt = GetScalarValue<int64_t>(input_args[kInputIndex4]->GetValue());
  if (!axis_opt.has_value()) {
    output_shape = ShapeVector(in_shape.size() + 1, abstract::Shape::kShapeDimAny);
  } else {
    int64_t axis_value = axis_opt.value();
    auto in_shape_size = SizeToLong(in_shape.size());
    MS_CHECK_VALUE(axis_value >= -1 && axis_value <= in_shape_size,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("axis value", axis_value, kIncludeBoth,
                                                               {-1, in_shape_size}, primitive));
    if (axis_value >= 0) {
      (void)output_shape.insert(output_shape.begin() + axis_value, depth_value);
    } else {
      output_shape.push_back(depth_value);
    }
  }
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr OneHotExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto tensor_type = input_args[kInputIndex0]->GetType();
  return tensor_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
