/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/batch_norm.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(BatchNorm, BaseOperator);
MIND_API_OPERATOR_IMPL(BatchNormWithActivation, BatchNorm);
MIND_API_OPERATOR_IMPL(BatchNormWithAddAndActivation, BatchNorm);
void BatchNorm::Init(const bool is_training, const float epsilon, const float momentum, const Format &format) {
  set_is_training(is_training);
  set_epsilon(epsilon);
  set_format(format);
  set_momentum(momentum);
}

void BatchNorm::set_is_training(const bool is_training) {
  (void)this->AddAttr(kIsTraining, api::MakeValue(is_training));
}

void BatchNorm::set_epsilon(const float epsilon) {
  CheckAndConvertUtils::CheckInRange<float>(kEpsilon, epsilon, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon));
}

void BatchNorm::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

void BatchNorm::set_momentum(const float momentun) {
  CheckAndConvertUtils::CheckInRange<float>(kMomentum, momentun, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kMomentum, api::MakeValue(momentun));
}

float BatchNorm::get_momentum() const {
  auto value_ptr = GetAttr(kMomentum);
  return GetValue<float>(value_ptr);
}

bool BatchNorm::get_is_training() const {
  auto value_ptr = GetAttr(kIsTraining);
  return GetValue<bool>(value_ptr);
}

float BatchNorm::get_epsilon() const {
  auto value_ptr = GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}

Format BatchNorm::get_format() const {
  auto value_ptr = GetAttr(kFormat);
  MS_EXCEPTION_IF_NULL(value_ptr);
  if (!value_ptr->isa<mindspore::api::StringImm>()) {
    return Format(GetValue<int64_t>(value_ptr));
  }
  static const std::map<std::string, int64_t> valid_dataformat = {
    {"NHWC", Format::NHWC},
    {"NCHW", Format::NCHW},
  };
  auto attr_value_str = GetValue<std::string>(value_ptr);
  (void)std::transform(attr_value_str.begin(), attr_value_str.end(), attr_value_str.begin(), toupper);
  auto iter = valid_dataformat.find(attr_value_str);
  if (iter == valid_dataformat.end()) {
    MS_LOG(EXCEPTION) << "for BatchNorm, Invalid format " << attr_value_str << ", use NHWC or NCHW";
  }
  return Format(iter->second);
}

namespace {
bool MeanAndVarianceValid(const std::vector<AbstractBasePtr> &input_args) {
  std::vector<int> params_ids = {3, 4};
  size_t valid_param = 0;
  for (auto idx : params_ids) {
    auto type = input_args[idx]->BuildType();
    if (type->isa<TensorType>()) {
      auto tensor_type = type->cast<TensorTypePtr>();
      auto element = tensor_type->element();
      valid_param += element->type_id() != kMetaTypeNone ? 1 : 0;
    }
  }
  return valid_param == params_ids.size();
}

std::string get_format_in_infer(const PrimitivePtr &primitive) {
  auto format_ptr = primitive->GetAttr(kFormat);
  if (format_ptr->isa<StringImm>()) {
    return GetValue<std::string>(format_ptr);
  }
  auto format = Format(GetValue<int64_t>(format_ptr));
  return FormatEnumToString(format);
}
}  // namespace

class BatchNormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, 0, prim_name);
    auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
    auto scale_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(scale_shape_ptr)[kShape];
    auto bias_shape_ptr = input_args[kInputIndex2]->BuildShape();
    auto bias_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(bias_shape_ptr)[kShape];
    auto mean_shape_ptr = input_args[kInputIndex3]->BuildShape();
    auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mean_shape_ptr)[kShape];
    auto variance_shape_ptr = input_args[kInputIndex4]->BuildShape();
    auto variance_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(variance_shape_ptr)[kShape];

    auto is_training = GetValue<bool>(primitive->GetAttr(kIsTraining));

    (void)CheckAndConvertUtils::CheckInteger("rank of scale", SizeToLong(scale_shape.size()), kEqual, 1, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of bias", SizeToLong(bias_shape.size()), kEqual, 1, prim_name);

    if (!x_shape_ptr->IsDynamic() && !scale_shape_ptr->IsDynamic()) {
      // auto format = GetValue<std::string>(primitive->GetAttr(kFormat));
      (void)CheckAndConvertUtils::CheckInRange("rank of images", SizeToLong(x_shape.size()), kIncludeBoth, {2, 4},
                                               prim_name);
      auto format = get_format_in_infer(primitive);
      auto channel = format == "NHWC" ? x_shape.back() : x_shape[1];
      if (scale_shape[kInputIndex0] != channel) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', 'scale_dim0' and input channel should be equal, but got "
                                 << scale_shape[kInputIndex0] << " and " << channel << ".";
      }
    }

    if (!mean_shape_ptr->IsDynamic() && !variance_shape_ptr->IsDynamic() && !bias_shape_ptr->IsDynamic() &&
        !scale_shape_ptr->IsDynamic()) {
      if (scale_shape[0] != bias_shape[0]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', 'scale' and 'bias' should have the same size, but got "
                                 << scale_shape[0] << ", " << bias_shape[0] << '.';
      }
      if (MeanAndVarianceValid(input_args)) {
        (void)CheckAndConvertUtils::CheckInteger("rank of mean", SizeToLong(mean_shape.size()), kEqual, 1, prim_name);
        (void)CheckAndConvertUtils::CheckInteger("rank of variance", SizeToLong(variance_shape.size()), kEqual, 1,
                                                 prim_name);
        if (!is_training && (mean_shape[0] != variance_shape[0] || variance_shape[0] != scale_shape[0])) {
          MS_EXCEPTION(ValueError)
            << "For '" << prim_name
            << "', 'scale', 'bias', 'mean', and 'variance' should have the same size during training, but got "
            << scale_shape[0] << ", " << bias_shape[0] << mean_shape[0] << " and " << variance_shape[0] << ".";
        }
      }
    }

    abstract::ShapePtr y_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
    abstract::ShapePtr channel_shape_ptr = std::make_shared<abstract::Shape>(scale_shape);

    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      y_shape_ptr, channel_shape_ptr, channel_shape_ptr, channel_shape_ptr, channel_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, 0, prim_name);
    const std::set valid_types = {kFloat16, kFloat32};
    auto x_type = input_args[0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, valid_types, prim->name());

    std::map<std::string, TypePtr> types;
    auto scale_type = input_args[kInputIndex1]->BuildType();
    (void)types.emplace("scale", input_args[kInputIndex1]->BuildType());
    (void)types.emplace("bias", input_args[kInputIndex2]->BuildType());
    if (MeanAndVarianceValid(input_args)) {
      (void)types.emplace("mean", input_args[kInputIndex3]->BuildType());
      (void)types.emplace("variance", input_args[kInputIndex4]->BuildType());
    }
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, scale_type, scale_type, scale_type, scale_type});
  }
};

abstract::AbstractBasePtr BatchNormInferFunc(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  BatchNormInfer bn;
  auto type = bn.InferType(primitive, input_args);
  auto shape = bn.InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNorm, prim::kPrimBatchNorm, BatchNormInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNormWithActivation, prim::kPrimBatchNormWithActivation, BatchNormInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNormWithAddAndActivation, prim::kPrimBatchNormWithAddAndActivation,
                                 BatchNormInfer, false);
}  // namespace ops
}  // namespace mindspore
