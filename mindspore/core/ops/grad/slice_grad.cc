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

#include "ops/grad/slice_grad.h"
#include <set>
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SliceGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 4, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto input_shape = shape_map[kShape];
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr SliceGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto input_args_rank = SizeToLong(input_args.size());
  constexpr int64_t kInputNum = 4;
  (void)CheckAndConvertUtils::CheckInteger("slice_grad_prim_infer", input_args_rank, kEqual, kInputNum, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  MS_EXCEPTION_IF_NULL(input_args[1]);
  auto x_type_map = input_args[1]->BuildType();
  MS_EXCEPTION_IF_NULL(x_type_map);
  auto x_dtype = x_type_map->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_dtype);
  std::set<TypePtr> template_types = {kTensorType};
  return CheckAndConvertUtils::CheckTensorTypeValid("x_dtype", x_dtype, template_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(SliceGrad, BaseOperator);
AbstractBasePtr SliceGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SliceGradInferShape(primitive, input_args), SliceGradInferType(primitive, input_args));
}

REGISTER_PRIMITIVE_C(kNameSliceGrad, SliceGrad);
}  // namespace ops
}  // namespace mindspore
