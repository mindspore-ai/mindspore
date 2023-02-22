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

#include "ops/instance_norm_v2.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
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
constexpr size_t InstanceNormV2InXIndex = kInputIndex0;
constexpr size_t InstanceNormV2InGammaIndex = kInputIndex1;
constexpr size_t InstanceNormV2InBetaIndex = kInputIndex2;
constexpr size_t InstanceNormV2InMeanIndex = kInputIndex3;
constexpr size_t InstanceNormV2InVarianceIndex = kInputIndex4;

void InstanceNormV2InputShapeCheck(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x_shape_ptr = input_args[InstanceNormV2InXIndex]->BuildShape();
  auto gamma_shape_ptr = input_args[InstanceNormV2InGammaIndex]->BuildShape();
  auto beta_shape_ptr = input_args[InstanceNormV2InBetaIndex]->BuildShape();
  auto mean_shape_ptr = input_args[InstanceNormV2InMeanIndex]->BuildShape();
  auto variance_shape_ptr = input_args[InstanceNormV2InVarianceIndex]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("x", x_shape, op_name);

  const std::map<std::string, BaseShapePtr> shapes = {
    {"gamma", gamma_shape_ptr}, {"beta", beta_shape_ptr}, {"mean", mean_shape_ptr}, {"variance", variance_shape_ptr}};
  ShapeVector check_shape = x_shape;
  int64_t image_size = 0;
  constexpr int64_t kDimSizeOne = 1;
  size_t dim4 = static_cast<size_t>(kDim4);
  size_t dim5 = static_cast<size_t>(kDim5);
  if (x_shape.size() == dim4) {
    // data format NCHW
    check_shape[kFormatNCHWIndexH] = kDimSizeOne;
    check_shape[kFormatNCHWIndexW] = kDimSizeOne;
    image_size = x_shape[kFormatNCHWIndexH] * x_shape[kFormatNCHWIndexW];
  } else if (x_shape.size() == dim5) {
    // data format NC1HWC0
    check_shape[kFormatNC1HWC0IndexH] = kDimSizeOne;
    check_shape[kFormatNC1HWC0IndexW] = kDimSizeOne;
    image_size = x_shape[kFormatNC1HWC0IndexH] * x_shape[kFormatNC1HWC0IndexW];
  } else {
    MS_EXCEPTION(ValueError)
      << "For " << op_name
      << ", input x only support 4D and 5D with data format NCHW and NC1HWC0, but get input x with shape of "
      << x_shape.size() << "D.";
  }
  constexpr auto min_size = 1;
  if (image_size < min_size) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ",  Expected more than 1 value per instance, but get input x size "
                             << ShapeVectorToStr(x_shape) << " with " << image_size << " value per instance.";
  }
  (void)CheckAndConvertUtils::CheckTensorShapeSame(shapes, check_shape, op_name);
}

abstract::TupleShapePtr InstanceNormV2InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const auto input_x_shape_ptr = input_args[InstanceNormV2InXIndex]->BuildShape();
  const auto mean_shape_ptr = input_args[InstanceNormV2InMeanIndex]->BuildShape();
  const auto var_shape_ptr = input_args[InstanceNormV2InVarianceIndex]->BuildShape();
  if (input_x_shape_ptr->IsDynamic() || mean_shape_ptr->IsDynamic() || var_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{input_x_shape_ptr, mean_shape_ptr, var_shape_ptr});
  }
  InstanceNormV2InputShapeCheck(primitive, input_args);
  auto is_training_ptr = primitive->GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(is_training_ptr);
  auto momentum_ptr = primitive->GetAttr(kMomentum);
  MS_EXCEPTION_IF_NULL(momentum_ptr);
  auto momentum = GetValue<float>(momentum_ptr);
  constexpr float momentum_min = 0.0;
  constexpr float momentum_max = 1.0;
  auto momentum_range = std::make_pair(momentum_min, momentum_max);
  CheckAndConvertUtils::CheckInRange(kMomentum, momentum, kIncludeBoth, momentum_range, prim_name);
  auto epsilon_ptr = primitive->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(epsilon_ptr);
  auto epsilon = GetValue<float>(epsilon_ptr);
  // momentum_range is equal to epsilon_range, but epsilon_range just include left.
  CheckAndConvertUtils::CheckInRange(kEpsilon, epsilon, kIncludeLeft, momentum_range, prim_name);

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_x_shape_ptr)[kShape];
  auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mean_shape_ptr)[kShape];
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
  auto _x_shape_ptr_ = std::make_shared<abstract::Shape>(x_shape);
  auto _mean_shape_ptr_ = std::make_shared<abstract::Shape>(mean_shape);
  auto _var_shape_ptr_ = std::make_shared<abstract::Shape>(var_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{_x_shape_ptr_, _mean_shape_ptr_, _var_shape_ptr_});
}

TuplePtr InstanceNormV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  const auto prim_name = primitive->name();
  const auto input_x = input_args[InstanceNormV2InXIndex]->BuildType();
  const auto gamma = input_args[InstanceNormV2InGammaIndex]->BuildType();
  const auto beta = input_args[InstanceNormV2InBetaIndex]->BuildType();
  const auto mean = input_args[InstanceNormV2InMeanIndex]->BuildType();
  const auto variance = input_args[InstanceNormV2InVarianceIndex]->BuildType();

  (void)CheckAndConvertUtils::CheckTypeValid("input x", input_x, {kFloat16, kFloat32}, prim_name);
  const std::map<std::string, TypePtr> types = {
    {"gamma", gamma},
    {"beta", beta},
    {"mean", mean},
    {"variance", variance},
  };
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_x, gamma, gamma});
}
}  // namespace

AbstractBasePtr InstanceNormV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 5;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto type = InstanceNormV2InferType(primitive, input_args);
  auto shape = InstanceNormV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
MIND_API_OPERATOR_IMPL(InstanceNormV2, BaseOperator);

// AG means auto generated
class MIND_API AGInstanceNormV2Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return InstanceNormV2InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return InstanceNormV2InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return InstanceNormV2Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(InstanceNormV2, prim::kPrimInstanceNormV2, AGInstanceNormV2Infer, false);
}  // namespace ops
}  // namespace mindspore
