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

#include <memory>
#include <algorithm>
#include "ops/grad/batch_norm_grad.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr auto kInputNum = 6;
}
MIND_API_OPERATOR_IMPL(BatchNormGrad, BaseOperator);
MIND_API_OPERATOR_IMPL(BatchNormGradWithActivation, BatchNormGrad);
MIND_API_OPERATOR_IMPL(BatchNormGradWithAddAndActivation, BatchNormGrad);
void BatchNormGrad::Init(const bool is_training, const float epsilon, const Format &format,
                         const std::string &inplace_algo) {
  this->set_is_training(is_training);
  this->set_epsilon(epsilon);
  this->set_format(format);
  this->set_inplace_algo(inplace_algo);
}

void BatchNormGrad::set_epsilon(const float epsilon) { (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon)); }

float BatchNormGrad::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<float>(value_ptr);
}

void BatchNormGrad::set_is_training(const bool is_training) {
  (void)this->AddAttr(kIsTraining, api::MakeValue(is_training));
}

bool BatchNormGrad::get_is_training() const {
  auto value_ptr = this->GetAttr(kIsTraining);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}

void BatchNormGrad::set_format(const Format &format) {
  int64_t f = format;
  (void)this->AddAttr(kFormat, api::MakeValue(f));
}

Format BatchNormGrad::get_format() const {
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
    MS_LOG(EXCEPTION) << "for BatchNormGrad, Invalid format " << attr_value_str << ", use NHWC or NCHW";
  }
  return Format(iter->second);
}

std::string BatchNormGrad::get_inplace_algo() const {
  auto value_ptr = GetAttr(kInplaceAlgo);
  if (value_ptr == nullptr) {
    return "cover";
  }
  return GetValue<std::string>(value_ptr);
}

void BatchNormGrad::set_inplace_algo(const std::string &inplace_algo) {
  (void)this->AddAttr(kInplaceAlgo, api::MakeValue(inplace_algo));
}

class BatchNormGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto x_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
    if (!IsDynamicRank(x_shape)) {
      (void)CheckAndConvertUtils::CheckInRange("rank of x", SizeToLong(x_shape.size()), kIncludeBoth, {2, 4},
                                               prim_name);
    }
    auto scale_shape_ptr = input_args[kInputIndex2]->BuildShape();
    if (prim_name == kNameBatchNormGradWithAddAndActivation) {
      return std::make_shared<abstract::TupleShape>(
        std::vector<abstract::BaseShapePtr>{x_shape_ptr, scale_shape_ptr, scale_shape_ptr, x_shape_ptr});
    }
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{x_shape_ptr, scale_shape_ptr, scale_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, prim_name);
    auto x_type_ptr = input_args[kInputIndex1]->BuildType();
    auto scale_type_ptr = input_args[kInputIndex2]->BuildType();
    if (prim_name == kNameBatchNormGradWithAddAndActivation) {
      return std::make_shared<Tuple>(std::vector<TypePtr>{x_type_ptr, scale_type_ptr, scale_type_ptr, x_type_ptr});
    }
    return std::make_shared<Tuple>(std::vector<TypePtr>{x_type_ptr, scale_type_ptr, scale_type_ptr});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNormGrad, prim::kPrimBatchNormGrad, BatchNormGradInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNormGradWithActivation, prim::kPrimBatchNormGradWithActivation,
                                 BatchNormGradInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(BatchNormGradWithAddAndActivation, prim::kPrimBatchNormGradWithAddAndActivation,
                                 BatchNormGradInfer, false);
}  // namespace ops
}  // namespace mindspore
