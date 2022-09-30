/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "ops/grad/bias_add_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
std::vector<int64_t> GetFormatShape(const int64_t &format, const std::vector<int64_t> &input_shape) {
  std::vector<int64_t> output_shape;
  if (format == NHWC) {
    output_shape.push_back(input_shape.back());
  } else {
    output_shape.push_back(input_shape[1]);
  }
  return output_shape;
}
abstract::ShapePtr BiasAddGradInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamic(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  int64_t format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  auto out_shape = GetFormatShape(format, input_shape);
  return std::make_shared<abstract::Shape>(out_shape);
}
TypePtr BiasAddGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckInteger("BiasAddGrad infer", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  MS_EXCEPTION_IF_NULL(input_args[0]);
  auto x_type_map = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_map);
  auto x_type = x_type_map->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_type);
  std::set<TypePtr> valid_x_type = {kTensorType};
  return CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_x_type, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(BiasAddGrad, BaseOperator);
AbstractBasePtr BiasAddGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(BiasAddGradInferShape(primitive, input_args),
                                BiasAddGradInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BiasAddGrad, prim::kPrimBiasAddGrad, BiasAddGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
