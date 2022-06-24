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
#include "ops/left_shift.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <set>

#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LeftShiftInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t max_dim = 8;
  auto in_shape_x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto in_shape_y = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("The dimension of LeftShift input", SizeToLong(in_shape_x.size()), kLessThan,
                                           max_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("The dimension of LeftShift input", SizeToLong(in_shape_y.size()), kLessThan,
                                           max_dim, prim_name);
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr LeftShiftInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  (void)abstract::CheckDtypeSame(prim_name, x, y);
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  if (!input_type->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "The " << prim_name << "'s"
                            << " input must be tensor type but got " << input_type->ToString();
  }
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, prim_name);
  return input_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(LeftShift, BaseOperator);
AbstractBasePtr LeftShiftInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = LeftShiftInferType(primitive, input_args);
  auto infer_shape = LeftShiftInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(LeftShift, prim::kPrimLeftShift, LeftShiftInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
