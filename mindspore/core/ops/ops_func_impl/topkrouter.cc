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

#include "ops/ops_func_impl/topkrouter.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {

BaseShapePtr TopKRouterFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args) const {
  // inputs (x, capacity, expert_num)
  const int64_t InputNumber = 3;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input1_shape_ptr = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(input1_shape_ptr);
  auto input3_shape_ptr = input_args[kInputIndex2]->GetShape();
  MS_EXCEPTION_IF_NULL(input3_shape_ptr);
  if (input1_shape_ptr->IsDynamic() || input3_shape_ptr->IsDynamic() || !IsValueKnown(input_args[kInputIndex2])) {
    MS_EXCEPTION(ValueError) << "For TopKRouter ops the input x and expert_num must be static shape";
  }

  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto expert_value = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  auto expert_num = expert_value.value();

  // check x shape
  auto x_rank = SizeToLong(input_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", x_rank, kEqual, InputNumber, prim_name);

  abstract::ShapePtr combine_shape_ptr;
  ShapeVector combine_shape = {input_shape};
  combine_shape_ptr = std::make_shared<abstract::Shape>(combine_shape);
  abstract::ShapePtr dispatch_shape_ptr;
  if (!IsValueKnown(input_args[kInputIndex1])) {
    ShapeVector dispatch_shape = {input_shape[0], expert_num, abstract::Shape::kShapeDimAny};
    dispatch_shape_ptr = std::make_shared<abstract::Shape>(dispatch_shape);
    std::vector<abstract::BaseShapePtr> shape_list = {dispatch_shape_ptr, combine_shape_ptr};
    return std::make_shared<abstract::TupleShape>(shape_list);
  }

  auto capacity_value = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto capacity = capacity_value.value();
  ShapeVector dispatch_shape = {input_shape[0], expert_num, capacity};  // rely on expert_num, capacity
  dispatch_shape_ptr = std::make_shared<abstract::Shape>(dispatch_shape);
  std::vector<abstract::BaseShapePtr> shape_list = {dispatch_shape_ptr, combine_shape_ptr};
  return std::make_shared<abstract::TupleShape>(shape_list);
}

TypePtr TopKRouterFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  std::set<TypePtr> scala_valid_types{kInt64};
  auto x_type = input_args[kInputIndex0]->GetType();
  auto capacity_type = input_args[kInputIndex1]->GetType();
  auto expert_num_type = input_args[kInputIndex2]->GetType();

  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("capacity", capacity_type, scala_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("exprt_num", expert_num_type, scala_valid_types, prim_name);
  std::vector<TypePtr> type_list = {x_type, x_type};
  return std::make_shared<Tuple>(type_list);
}

}  // namespace ops
}  // namespace mindspore
