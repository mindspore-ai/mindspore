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
#include "ops/rightshift.h"
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
abstract::ShapePtr RightShiftInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t max_dim = 8;
  auto in_shape_x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto in_shape_y = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("The dimension of RightShift input", SizeToLong(in_shape_x.size()),
                                           kLessThan, max_dim, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("The dimension of RightShift input", SizeToLong(in_shape_y.size()),
                                           kLessThan, max_dim, prim_name);
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr RightShiftInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto y_type = input_args[kInputIndex1]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type);
  MS_EXCEPTION_IF_NULL(y_type);
  std::map<std::string, TypePtr> types;
  types.insert({"input_x", x_type});
  types.insert({"input_y", y_type});
  const std::set<TypePtr> valid_types = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(RightShift, BaseOperator);
AbstractBasePtr RightShiftInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = RightShiftInferType(primitive, input_args);
  auto infer_shape = RightShiftInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(RightShift, prim::kPrimRightShift, RightShiftInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
