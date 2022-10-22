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

#include "ops/grad/max_unpool3d_grad.h"
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MaxUnpool3DGradInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShapeTrack())[kShape];
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShapeTrack())[kShape];
  auto argmax_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShapeTrack())[kShape];

  if (IsDynamic(in_shape)) {
    return std::make_shared<abstract::Shape>(in_shape);
  }

  (void)CheckAndConvertUtils::CheckInteger("x_rank", SizeToLong(in_shape.size()), kEqual, SizeToLong(kDim5), op_name);
  (void)CheckAndConvertUtils::CheckInteger("grads_rank", SizeToLong(grads_shape.size()), kEqual, SizeToLong(kDim5),
                                           op_name);
  (void)CheckAndConvertUtils::CheckInteger("argmax_rank", SizeToLong(argmax_shape.size()), kEqual, SizeToLong(kDim5),
                                           op_name);
  CheckAndConvertUtils::Check("x_shape", in_shape, kEqual, argmax_shape, op_name, ValueError);
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x1_shape);
}

TypePtr MaxUnpool3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> argmax_valid_types = {kInt32, kInt64};
  auto input_x_type = input_args[kInputIndex0]->BuildType();
  auto grads_type = input_args[kInputIndex1]->BuildType();
  auto argmax_type = input_args[kInputIndex2]->BuildType();
  auto output_type =
    CheckAndConvertUtils::CheckTensorTypeValid("x", input_x_type, common_valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grads", grads_type, common_valid_types, primitive->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax", argmax_type, argmax_valid_types, primitive->name());
  return output_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(MaxUnpool3DGrad, BaseOperator);
AbstractBasePtr MaxUnpool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto infer_type = MaxUnpool3DGradInferType(primitive, input_args);
  auto infer_shape = MaxUnpool3DGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
std::string MaxUnpool3DGrad::get_format() const { return GetValue<std::string>(GetAttr(kFormat)); }
REGISTER_PRIMITIVE_EVAL_IMPL(MaxUnpool3DGrad, prim::kPrimMaxUnpool3DGrad, MaxUnpool3DGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
