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

#include "ops/grad/instance_norm_v2_grad.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInstanceNomrV2GradInDyIndex = kInputIndex0;
constexpr size_t kInstanceNomrV2GradInXIndex = kInputIndex1;
constexpr size_t kInstanceNomrV2GradInGammaIndex = kInputIndex2;
constexpr size_t kInstanceNomrV2GradInMeanIndex = kInputIndex3;
constexpr size_t kInstanceNomrV2GradInVarianceIndex = kInputIndex4;
constexpr size_t kInstanceNomrV2GradInSaveMeanIndex = kInputIndex5;
constexpr size_t kInstanceNomrV2GradInSaveVarianceIndex = kInputIndex6;

void InstanceNormV2GradInputShapeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto dy_shape_ptr = input_args[kInstanceNomrV2GradInDyIndex]->BuildShape();
  auto x_shape_ptr = input_args[kInstanceNomrV2GradInXIndex]->BuildShape();
  auto gamma_shape_ptr = input_args[kInstanceNomrV2GradInGammaIndex]->BuildShape();
  auto mean_shape_ptr = input_args[kInstanceNomrV2GradInMeanIndex]->BuildShape();
  auto variance_shape_ptr = input_args[kInstanceNomrV2GradInVarianceIndex]->BuildShape();
  auto save_mean_shape_ptr = input_args[kInstanceNomrV2GradInSaveMeanIndex]->BuildShape();
  auto save_variance_shape_ptr = input_args[kInstanceNomrV2GradInSaveVarianceIndex]->BuildShape();
  auto dy_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dy_shape_ptr)[kShape];
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("dy", dy_shape, op_name);
  (void)CheckAndConvertUtils::CheckTensorShapeSame({{"x", x_shape_ptr}}, dy_shape, op_name);

  const std::map<std::string, BaseShapePtr> shapes = {{"gamma", gamma_shape_ptr},
                                                      {"mean", mean_shape_ptr},
                                                      {"variance", variance_shape_ptr},
                                                      {"save_mean", save_mean_shape_ptr},
                                                      {"save_variance", save_variance_shape_ptr}};
  ShapeVector check_shape = dy_shape;
  int64_t image_size = 0;
  constexpr int64_t kDimSizeOne = 1;
  size_t dim4 = static_cast<size_t>(kDim4);
  size_t dim5 = static_cast<size_t>(kDim5);
  if (dy_shape.size() == dim4) {
    // data format NCHW
    check_shape[kFormatNCHWIndexH] = kDimSizeOne;
    check_shape[kFormatNCHWIndexW] = kDimSizeOne;
    image_size = dy_shape[kFormatNCHWIndexH] * dy_shape[kFormatNCHWIndexW];
  } else if (dy_shape.size() == dim5) {
    // data format NC1HWC0
    check_shape[kFormatNC1HWC0IndexH] = kDimSizeOne;
    check_shape[kFormatNC1HWC0IndexW] = kDimSizeOne;
    image_size = dy_shape[kFormatNC1HWC0IndexH] * dy_shape[kFormatNC1HWC0IndexW];
  } else {
    MS_EXCEPTION(ValueError)
      << "For " << op_name
      << ", input dy only support 4D and 5D with data format NCHW and NC1HWC0, but get input dy with shape of "
      << dy_shape.size() << "D.";
  }
  constexpr auto min_size = 1;
  if (image_size < min_size) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ",  Expected more than 1 value per instance, but get dy size "
                             << ShapeVectorToStr(dy_shape) << " with " << image_size << " value per instance.";
  }
  (void)CheckAndConvertUtils::CheckTensorShapeSame(shapes, check_shape, op_name);
}

abstract::TupleShapePtr InstanceNormV2GradInferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const auto dy_shape_ptr = input_args[kInstanceNomrV2GradInDyIndex]->BuildShape();
  const auto gamma_shape_ptr = input_args[kInstanceNomrV2GradInGammaIndex]->BuildShape();
  if (dy_shape_ptr->IsDynamic() || gamma_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{dy_shape_ptr, gamma_shape_ptr, gamma_shape_ptr});
  }
  InstanceNormV2GradInputShapeCheck(primitive, input_args);
  auto is_training_ptr = primitive->GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(is_training_ptr);
  auto epsilon_ptr = primitive->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(epsilon_ptr);
  auto epsilon = GetValue<float>(epsilon_ptr);
  constexpr float epsilon_min = 0.0;
  constexpr float epsilon_max = 1.0;
  auto epsilon_range = std::make_pair(epsilon_min, epsilon_max);
  // momentum_range is equal to epsilon_range, but epsilon_range just include left.
  CheckAndConvertUtils::CheckInRange(kEpsilon, epsilon, kIncludeLeft, epsilon_range, prim_name);
  auto dx_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(dy_shape_ptr)[kShape];
  auto pd_gamma_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(gamma_shape_ptr)[kShape];
  auto pd_beta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(gamma_shape_ptr)[kShape];
  auto _dx_shape_ptr_ = std::make_shared<abstract::Shape>(dx_shape);
  auto _pd_gamma_shape_ptr_ = std::make_shared<abstract::Shape>(pd_gamma_shape);
  auto _pd_beta_shape_ptr_ = std::make_shared<abstract::Shape>(pd_beta_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{_dx_shape_ptr_, _pd_gamma_shape_ptr_, _pd_beta_shape_ptr_});
}

TuplePtr InstanceNormV2GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto prim_name = primitive->name();
  const auto dy = input_args[kInstanceNomrV2GradInDyIndex]->BuildType();
  const auto x = input_args[kInstanceNomrV2GradInXIndex]->BuildType();
  const auto gamma = input_args[kInstanceNomrV2GradInGammaIndex]->BuildType();
  const auto mean = input_args[kInstanceNomrV2GradInMeanIndex]->BuildType();
  const auto variance = input_args[kInstanceNomrV2GradInVarianceIndex]->BuildType();
  const auto save_mean = input_args[kInstanceNomrV2GradInSaveMeanIndex]->BuildType();
  const auto save_variance = input_args[kInstanceNomrV2GradInSaveVarianceIndex]->BuildType();

  (void)CheckAndConvertUtils::CheckTypeValid("input dy", dy, {kFloat16, kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("input x", x, {kFloat16, kFloat32}, prim_name);
  const std::map<std::string, TypePtr> types = {
    {"gamma", gamma},
    {"mean", mean},
    {"variance", variance},
    {"save_mean", save_mean},
    {"save_variance", save_variance},
  };
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{dy, gamma, gamma});
}
}  // namespace

AbstractBasePtr InstanceNormV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 7;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto type = InstanceNormV2GradInferType(primitive, input_args);
  auto shape = InstanceNormV2GradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
MIND_API_OPERATOR_IMPL(InstanceNormV2Grad, BaseOperator);

// AG means auto generated
class MIND_API AGInstanceNormV2GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InstanceNormV2GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return InstanceNormV2GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return InstanceNormV2GradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(InstanceNormV2Grad, prim::kPrimInstanceNormV2Grad, AGInstanceNormV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
