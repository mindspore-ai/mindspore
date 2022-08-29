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

#include "ops/erfinv.h"
#include <map>
#include <string>
#include <set>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ErfinvInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_map[kShape];
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr ErfinvInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto infer_type = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", infer_type, valid_types, prim->name());
  return infer_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Erfinv, BaseOperator);
AbstractBasePtr ErfinvInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input_x numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(ErfinvInferShape(primitive, input_args), ErfinvInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(Erfinv, prim::kPrimErfinv, ErfinvInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
