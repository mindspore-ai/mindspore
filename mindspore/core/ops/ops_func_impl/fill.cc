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

#include "ops/ops_func_impl/fill.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr FillFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto shape_v = GetArrayValue<int64_t>(input_args[kInputIndex0]);
  if (!shape_v.has_value()) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto shape = shape_v.value();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape_v->size(); i++) {
    if (shape.IsValueUnknown(i)) {
      output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    } else {
      int64_t shape_i = shape[i];
      MS_CHECK_VALUE(shape_i >= 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                     "the " + std::to_string(i) + "th dimension of input shape", shape_i, kGreaterEqual,
                                     0, primitive));
      output_shape.push_back(shape_i);
    }
  }

  return std::make_shared<abstract::TensorShape>(output_shape);
}

ShapeArray FillFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  if (input_values.size() != kSize3) {
    MS_EXCEPTION(ValueError) << primitive->name() << " should have" << kSize3 << "inputs. Please try other inputs";
  }
  const auto &size = input_values[kIndex0]->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(size);

  std::vector<int64_t> size_vector;
  for (const auto &value : size->value()) {
    const auto &size_element = value->cast<Int64ImmPtr>();
    MS_EXCEPTION_IF_NULL(size_element);
    size_vector.emplace_back(size_element->value());
  }
  ShapeVector output_shape;
  for (size_t i = 0; i < size_vector.size(); i++) {
    int64_t shape_i = size_vector[i];
    MS_CHECK_VALUE(shape_i >= 0,
                   CheckAndConvertUtils::FormatCheckIntegerMsg(
                     "the " + std::to_string(i) + "th dimension of input shape", shape_i, kGreaterEqual, 0, primitive));
    output_shape.push_back(shape_i);
  }

  return {output_shape};
}
}  // namespace ops
}  // namespace mindspore
