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
#include "ops/bessel_y0.h"

#include <algorithm>
#include <set>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BesselY0InferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  return std::make_shared<abstract::Shape>(in_shape);
}
TypePtr BesselY0InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[kInputIndex0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(BesselY0, BaseOperator);
AbstractBasePtr BesselY0Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kInputNum,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto infer_type = BesselY0InferType(primitive, input_args);
  auto infer_shape = BesselY0InferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(BesselY0, prim::kPrimBesselY0, BesselY0Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
