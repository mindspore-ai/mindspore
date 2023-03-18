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

#include "ops/scale_and_translate.h"

#include <set>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/named.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ScaleAndTranslateInferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto images_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto size_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto translation_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(images_shape) || IsDynamicRank(size_shape) || IsDynamicRank(scale_shape) ||
      IsDynamicRank(translation_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }
  // support dynamic shape
  if (IsDynamicShape(images_shape) || IsDynamicShape(size_shape) || IsDynamicShape(scale_shape) ||
      IsDynamicShape(translation_shape)) {
    return std::make_shared<abstract::Shape>(
      ShapeVector({abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                   abstract::Shape::kShapeDimAny}));
  }

  const int64_t kShapeSize = 1;
  const int64_t kElementsNumber = 2;
  const int64_t kImagesShapeSize = 4;
  auto images_shape_size = images_shape.size();
  auto size_shape_size = size_shape.size();
  auto scale_shape_size = scale_shape.size();
  auto translation_shape_size = translation_shape.size();
  // check images' rank must be 4
  (void)CheckAndConvertUtils::CheckInteger("images's rank'", SizeToLong(images_shape_size), kEqual, kImagesShapeSize,
                                           prim_name);
  // check size' rank must be 1, must have 2 elements
  (void)CheckAndConvertUtils::CheckInteger("size's rank'", SizeToLong(size_shape_size), kEqual, kShapeSize, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("size's elements'", size_shape[0], kEqual, kElementsNumber, prim_name);
  // check scale' rank must be 1, must have 2 elements
  (void)CheckAndConvertUtils::CheckInteger("scale's rank'", SizeToLong(scale_shape_size), kEqual, kShapeSize,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("scale's elements'", scale_shape[0], kEqual, kElementsNumber, prim_name);
  // check translation' rank must be 1, must have 2 elements
  (void)CheckAndConvertUtils::CheckInteger("translation's rank'", SizeToLong(translation_shape_size), kEqual,
                                           kShapeSize, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("translation's elements'", translation_shape[0], kEqual, kElementsNumber,
                                           prim_name);
  // check scale greater than zero
  auto scale_v = input_args[kInputIndex2]->BuildValue();
  MS_EXCEPTION_IF_NULL(scale_v);
  if (!scale_v->isa<ValueAny>() && !scale_v->isa<None>()) {
    if (scale_v == nullptr) {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the input argument[scale]"
                               << " value is nullptr.";
    }
    std::vector<float> scale_value;
    if (!scale_v->isa<tensor::Tensor>()) {
      MS_EXCEPTION(ValueError) << "For primitive[" << prim_name << "], the input argument[scale]"
                               << " must be a tensor, but got " << scale_v->ToString();
    }
    auto scale_tensor = scale_v->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(scale_tensor);
    size_t data_size = scale_tensor->DataSize();
    auto data_c = static_cast<float *>(scale_tensor->data_c());
    MS_EXCEPTION_IF_NULL(data_c);
    for (size_t i = 0; i < data_size; i++) {
      scale_value.push_back(static_cast<float>(*data_c));
      ++data_c;
    }
    (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("scale", scale_value, prim_name);
  }
  //  infer resized_images's shape
  auto size_v = input_args[kInputIndex1]->BuildValue();
  MS_EXCEPTION_IF_NULL(size_v);
  std::vector<int64_t> size_value;
  if (!size_v->isa<ValueAny>() && !size_v->isa<None>()) {
    size_value = CheckAndConvertUtils::CheckTensorIntValue("size", size_v, prim_name);
    // check scale greater than zero
    (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("size", size_value, prim_name);
    std::vector<int64_t> out_shape;
    (void)out_shape.emplace_back(images_shape[kInputIndex0]);
    (void)out_shape.emplace_back(size_value[kInputIndex0]);
    (void)out_shape.emplace_back(size_value[kInputIndex1]);
    (void)out_shape.emplace_back(images_shape[kInputIndex3]);
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> out_shape;
    (void)out_shape.emplace_back(images_shape[kInputIndex0]);
    (void)out_shape.emplace_back(-1);
    (void)out_shape.emplace_back(-1);
    (void)out_shape.emplace_back(images_shape[kInputIndex3]);
    return std::make_shared<abstract::Shape>(out_shape);
  }
}

TypePtr ScaleAndTranslateInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto images_type = input_args[kInputIndex0]->BuildType();
  auto size_type = input_args[kInputIndex1]->BuildType();
  auto scale_type = input_args[kInputIndex2]->BuildType();
  auto translation_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> images_valid_types = {kInt8, kInt16, kInt32, kInt64, kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> size_valid_types = {kInt32};
  const std::set<TypePtr> valid_types = {kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("images", images_type, images_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("size", size_type, size_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("scale", scale_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("translation", translation_type, valid_types, prim_name);
  return std::make_shared<TensorType>(kFloat32);
}
}  // namespace

MIND_API_OPERATOR_IMPL(ScaleAndTranslate, BaseOperator);
void ScaleAndTranslate::Init(const std::string kernel_type, const bool antialias) {
  set_kernel_type(kernel_type);
  set_antialias(antialias);
}

void ScaleAndTranslate::set_kernel_type(const std::string kernel_type) {
  (void)this->AddAttr(kKernelType, api::MakeValue(kernel_type));
}

void ScaleAndTranslate::set_antialias(const bool antialias) {
  (void)this->AddAttr(kAntialias, api::MakeValue(antialias));
}

std::string ScaleAndTranslate::get_kernel_type() const { return GetValue<std::string>(GetAttr(kKernelType)); }

bool ScaleAndTranslate::get_antialias() const {
  auto value_ptr = GetAttr(kAntialias);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr ScaleAndTranslateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  auto infer_type = ScaleAndTranslateInferType(primitive, input_args);
  auto infer_shape = ScaleAndTranslateInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGScaleAndTranslateInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ScaleAndTranslateInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ScaleAndTranslateInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ScaleAndTranslateInfer(engine, primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2}; }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ScaleAndTranslate, prim::kPrimScaleAndTranslate, AGScaleAndTranslateInfer, false);
}  // namespace ops
}  // namespace mindspore
