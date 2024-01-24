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

#include "ops/ops_func_impl/correlate.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void InitArray() {}
BaseShapePtr CorrelateFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto input_a_shape_ptr = input_args[kIndex0]->GetShape();
  auto input_v_shape_ptr = input_args[kIndex1]->GetShape();
  auto input_a_shape = input_a_shape_ptr->GetShapeVector();
  auto input_v_shape = input_v_shape_ptr->GetShapeVector();
  if (IsDynamic(input_a_shape) || IsDynamic(input_v_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});
  }

  auto a_rank = input_a_shape.size();
  auto v_rank = input_v_shape.size();
  if (a_rank != 1 || v_rank != 1) {
    MS_EXCEPTION(ValueError) << "'" << primitive->name() << "' only support 1-dimensional inputs , but got a at "
                             << a_rank << "-dimensional and got v at " << v_rank << "-dimensional";
  }

  int a_len = input_a_shape[0];
  int v_len = input_v_shape[0];
  if (a_len == 0 || v_len == 0) {
    MS_EXCEPTION(ValueError) << "all inputs of '" << primitive->name() << "' cannot be empty , got a at ( " << a_len
                             << ") and got v at (" << v_len << ")";
  }

  int long_len = a_len < v_len ? v_len : a_len;
  int short_len = a_len < v_len ? a_len : v_len;
  int out_len = 0;
  auto mode_v = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  mindspore::PadMode mode_type = static_cast<mindspore::PadMode>(mode_v.value_or(-1));
  if (mode_type == mindspore::PadMode::VALID)
    out_len = long_len - short_len + 1;
  else if (mode_type == mindspore::PadMode::SAME)
    out_len = long_len;
  else if (mode_type == mindspore::PadMode::FULL)
    out_len = long_len + short_len - 1;
  else
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', the mode should be one of [valid, same, full], but got " << mode_type;
  return std::make_shared<abstract::Shape>(ShapeVector{out_len});
}

TypePtr CorrelateFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto input_a_type = input_args[kInputIndex0]->GetType();
  auto input_a_type_id = input_a_type->cast<TensorTypePtr>()->element()->type_id();
  auto input_v_type = input_args[kInputIndex1]->GetType();
  auto input_v_type_id = input_v_type->cast<TensorTypePtr>()->element()->type_id();
  if (input_a_type_id != input_v_type_id) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "' the type of a and v must be same, but got type of a is different from that of v! ";
  }

  static const std::vector<TypeId> type_to_float32 = {
    kNumberTypeInt8,
    kNumberTypeInt16,
    kNumberTypeInt32,
  };
  static const std::vector<TypeId> type_to_float64 = {kNumberTypeInt64};
  bool is_type_to_float32 =
    std::any_of(type_to_float32.begin(), type_to_float32.end(),
                [&input_a_type_id](const TypeId &type_id) { return input_a_type_id == type_id; });
  bool is_type_to_float64 =
    std::any_of(type_to_float64.begin(), type_to_float64.end(),
                [&input_a_type_id](const TypeId &type_id) { return input_a_type_id == type_id; });

  if (is_type_to_float32) {
    return std::make_shared<TensorType>(kFloat32);
  } else if (is_type_to_float64) {
    return std::make_shared<TensorType>(kFloat64);
  } else {
    return input_a_type->Clone();
  }
}

}  // namespace ops
}  // namespace mindspore
