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

#include "ops/grad/scale_and_translate_grad.h"

#include <algorithm>
#include <set>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScaleAndTranslateGradInferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto grads_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto original_image_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto translation_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(grads_shape) || IsDynamicRank(original_image_shape) || IsDynamicRank(scale_shape) ||
      IsDynamicRank(translation_shape)) {
    return std::make_shared<abstract::Shape>(original_image_shape);
  }
  // support dynamic shape
  if (IsDynamicShape(grads_shape) || IsDynamicShape(original_image_shape) || IsDynamicShape(scale_shape) ||
      IsDynamicShape(translation_shape)) {
    return std::make_shared<abstract::Shape>(original_image_shape);
  }
  const int64_t kShapeSize1 = 1;
  const int64_t kShapeSize2 = 4;
  const int64_t kElementsNumber = 2;
  // check grads rank'4
  (void)CheckAndConvertUtils::CheckInteger("grads's rank'", SizeToLong(grads_shape.size()), kEqual, kShapeSize2,
                                           prim_name);
  // check original_image's rank 4
  (void)CheckAndConvertUtils::CheckInteger("original_image's rank'", SizeToLong(original_image_shape.size()), kEqual,
                                           kShapeSize2, prim_name);
  // check scale' rank must be 1, must have 2 elements
  (void)CheckAndConvertUtils::CheckInteger("scale's rank'", SizeToLong(scale_shape.size()), kEqual, kShapeSize1,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("scale's elements'", scale_shape[0], kEqual, kElementsNumber, prim_name);
  // check translation' rank must be 1, must have 2 elements
  (void)CheckAndConvertUtils::CheckInteger("translation's rank'", SizeToLong(translation_shape.size()), kEqual,
                                           kShapeSize1, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("translation's elements'", translation_shape[0], kEqual, kElementsNumber,
                                           prim_name);
  //  infer output shape
  if (grads_shape[0] != original_image_shape[0]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of grads_shape[0] is " << grads_shape[0]
                             << ", but the shape of original_image_shape[0] is " << original_image_shape[0]
                             << ". The first dimension of the shape of grads_shape "
                             << "must be equal to that of original_image_shape.";
  }
  if (grads_shape[kInputIndex3] != original_image_shape[kInputIndex3]) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the shape of grads_shape[3] is "
                             << grads_shape[kInputIndex3] << ", but the shape of original_image_shape[3] is "
                             << original_image_shape[kInputIndex3]
                             << ". The third dimension of the shape of grads_shape "
                             << "must be equal to that of original_image_shape.";
  }
  return std::make_shared<abstract::Shape>(original_image_shape);
}

TypePtr ScaleAndTranslateGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto grads_type = input_args[kInputIndex0]->BuildType();
  auto original_image_type = input_args[kInputIndex1]->BuildType();
  auto scale_type = input_args[kInputIndex2]->BuildType();
  auto translation_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat32};
  std::map<std::string, TypePtr> args;
  // origin_image have the same type as grads
  (void)args.emplace("grads", grads_type);
  (void)args.emplace("original_image", original_image_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scale", scale_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("translation", translation_type, valid_types, prim_name);
  return grads_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(ScaleAndTranslateGrad, BaseOperator);
void ScaleAndTranslateGrad::Init(const std::string kernel_type, const bool antialias) {
  set_kernel_type(kernel_type);
  set_antialias(antialias);
}

void ScaleAndTranslateGrad::set_kernel_type(const std::string kernel_type) {
  (void)this->AddAttr(kKernelType, api::MakeValue(kernel_type));
}

void ScaleAndTranslateGrad::set_antialias(const bool antialias) {
  (void)this->AddAttr(kAntialias, api::MakeValue(antialias));
}

std::string ScaleAndTranslateGrad::get_kernel_type() const { return GetValue<std::string>(GetAttr(kKernelType)); }

bool ScaleAndTranslateGrad::get_antialias() const {
  auto value_ptr = GetAttr(kAntialias);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr ScaleAndTranslateGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  auto infer_type = ScaleAndTranslateGradInferType(primitive, input_args);
  auto infer_shape = ScaleAndTranslateGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGScaleAndTranslateGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ScaleAndTranslateGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ScaleAndTranslateGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ScaleAndTranslateGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScaleAndTranslateGrad, prim::kPrimScaleAndTranslateGrad, AGScaleAndTranslateGradInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
