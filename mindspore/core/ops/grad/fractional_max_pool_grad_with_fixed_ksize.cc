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

#include "ops/grad/fractional_max_pool_grad_with_fixed_ksize.h"

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputsIndex0 = 0;
constexpr size_t kInputsIndex1 = 1;
constexpr size_t kInputsIndex2 = 2;
constexpr size_t kInputsDimSize = 4;
constexpr size_t kInputIndexN = 0;
constexpr size_t kInputIndexC = 1;

abstract::ShapePtr FractionalMaxPoolGradWithFixedKsizeInferShape(const PrimitivePtr &primitive,
                                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto data_format = GetValue<std::string>(primitive->GetAttr(kFormat));
  if (data_format != "NCHW") {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, attr data_format must be NCHW.";
  }

  auto origin_input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputsIndex0]->GetShapeTrack())[kShape];
  auto out_backprop_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputsIndex1]->GetShapeTrack())[kShape];
  auto argmax_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputsIndex2]->GetShapeTrack())[kShape];
  // dynamic rank
  if (IsDynamicRank(origin_input_shape) || IsDynamicRank(out_backprop_shape) || IsDynamicRank(argmax_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  // dynamic shape
  if (IsDynamic(origin_input_shape) || IsDynamic(out_backprop_shape) || IsDynamic(argmax_shape)) {
    if (!IsDynamic(origin_input_shape)) {
      return std::make_shared<abstract::Shape>(origin_input_shape);
    }
    ShapeVector output_shape_dyn;
    for (size_t i = 0; i < origin_input_shape.size(); ++i) {
      output_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(output_shape_dyn);
  }
  if (origin_input_shape.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the dimension of origin_input must be 4.";
  }
  if (out_backprop_shape.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the dimension of out_backprop must be 4.";
  }
  if (argmax_shape.size() != kInputsDimSize) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the dimension of argmax must be 4.";
  }

  for (size_t i = 0; i < kInputsDimSize; i++) {
    if (out_backprop_shape[i] != argmax_shape[i]) {
      MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, out_backprop and argmax must have "
                               << "the same shape.";
    }
  }

  if (origin_input_shape[kInputIndexN] != out_backprop_shape[kInputIndexN]) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the first dimension size of three inputs "
                             << "must be equal.";
  }
  if (origin_input_shape[kInputIndexC] != out_backprop_shape[kInputIndexC]) {
    MS_EXCEPTION(ValueError) << "For FractionalMaxPoolGradWithFixedKsize, the second dimension size of three inputs "
                             << "must be equal.";
  }

  return std::make_shared<abstract::Shape>(origin_input_shape);
}

TypePtr FractionalMaxPoolGradWithFixedKsizeInferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();

  const std::set<TypePtr> origin_input_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("origin_input dtype", input_args[kInputsIndex0]->BuildType(),
                                                   origin_input_valid_types, prim_name);

  const std::set<TypePtr> out_backprop_valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  auto y_dtype = CheckAndConvertUtils::CheckTensorTypeValid(
    "out_backprop dtype", input_args[kInputsIndex1]->BuildType(), out_backprop_valid_types, prim_name);

  const std::set<TypePtr> argmax_valid_types = {kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("argmax dtype", input_args[kInputsIndex2]->BuildType(),
                                                   argmax_valid_types, prim_name);

  return std::make_shared<TensorType>(y_dtype);
}
}  // namespace

MIND_API_OPERATOR_IMPL(FractionalMaxPoolGradWithFixedKsize, BaseOperator);
AbstractBasePtr FractionalMaxPoolGradWithFixedKsizeInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t inputs_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, inputs_num, primitive->name());

  auto infer_shape = FractionalMaxPoolGradWithFixedKsizeInferShape(primitive, input_args);
  auto infer_type = FractionalMaxPoolGradWithFixedKsizeInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void FractionalMaxPoolGradWithFixedKsize::Init(const std::string data_format) { set_data_format(data_format); }

void FractionalMaxPoolGradWithFixedKsize::set_data_format(const std::string data_format) {
  (void)this->AddAttr(kFormat, api::MakeValue(data_format));
}

std::string FractionalMaxPoolGradWithFixedKsize::get_data_format() const {
  return GetValue<std::string>(GetAttr(kFormat));
}

REGISTER_PRIMITIVE_EVAL_IMPL(FractionalMaxPoolGradWithFixedKsize, prim::kPrimFractionalMaxPoolGradWithFixedKsize,
                             FractionalMaxPoolGradWithFixedKsizeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
