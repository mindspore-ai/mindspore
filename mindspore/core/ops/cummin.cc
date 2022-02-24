/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <set>
#include "abstract/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "ops/cummin.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr CumminInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto y_shape = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, 0);
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto y_rank = x_shape.size();
  const int64_t min_dim = 0;
  (void)CheckAndConvertUtils::CheckInteger("the rank of input", SizeToLong(x_shape.size()), kGreaterThan, min_dim,
                                           prim_name);
  int64_t axis = GetValue<int64_t>(primitive->GetAttr("axis"));
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-y_rank, y_rank - 1}, prim_name);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_shape, y_shape});
}

TuplePtr CumminInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  MS_EXCEPTION_IF_NULL(input_args[0]);
  const std::set<TypePtr> valid_types = {kFloat32, kFloat16, kInt32, kInt8, kUInt8};
  auto x_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim_name);
  TypePtr argmin_type = kInt32;
  return std::make_shared<Tuple>(std::vector{x_type, argmin_type});
}
}  // namespace

AbstractBasePtr CumminInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(primitive);
  auto type = CumminInferType(primitive, input_args);
  auto shape = CumminInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Cummin, prim::kPrimCummin, CumminInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
