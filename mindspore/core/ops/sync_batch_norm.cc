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

#include "ops/sync_batch_norm.h"

#include <map>
#include <set>
#include <string>
#include <utility>

#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kSyncBatchNormInputNum = 5;

void CheckSyncBatchNormInputNum(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  if (input_args.empty()) {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kSyncBatchNormInputNum, prim_name);
    return;
  }

  // the inputs has U
  if (!input_args.back()->isa<abstract::AbstractMonad>()) {
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kSyncBatchNormInputNum, prim_name);
    return;
  }

  // the inputs has not U
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size() - 1), kEqual,
                                           kSyncBatchNormInputNum, prim_name);
  for (size_t index = 0; index < input_args.size(); index++) {
    if (input_args[index] == nullptr) {
      MS_EXCEPTION(ValueError) << "The " << index << "'s input of " << prim_name << " is nullptr.";
    }
  }
}

TuplePtr SyncBatchNormInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  CheckSyncBatchNormInputNum(prim, input_args);
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto x_dtype = input_args[0]->BuildType();
  auto scale_dtype = input_args[1]->BuildType();
  auto bias_dtype = input_args[2]->BuildType();
  auto mean_dtype = input_args[3]->BuildType();
  auto variance_dtype = input_args[4]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, valid_types, prim_name);
  // scale and bias must have the same type.
  std::map<std::string, TypePtr> args1;
  (void)args1.insert(std::make_pair("scale", scale_dtype));
  (void)args1.insert(std::make_pair("bias", bias_dtype));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args1, valid_types, prim_name);
  // mean and variance must have the same type.
  std::map<std::string, TypePtr> args2;
  (void)args2.insert(std::make_pair("mean", mean_dtype));
  (void)args2.insert(std::make_pair("variance", variance_dtype));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args2, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, scale_dtype, scale_dtype, mean_dtype, mean_dtype});
}

abstract::TupleShapePtr SyncBatchNormInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  CheckSyncBatchNormInputNum(primitive, input_args);
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[0]->BuildShape();
  auto scale_shape_ptr = input_args[1]->BuildShape();
  auto bias_shape_ptr = input_args[2]->BuildShape();
  auto mean_shape_ptr = input_args[3]->BuildShape();
  auto variance_shape_ptr = input_args[4]->BuildShape();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  auto scale_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(scale_shape_ptr)[kShape];
  auto bias_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(bias_shape_ptr)[kShape];
  auto mean_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mean_shape_ptr)[kShape];
  auto variance_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(variance_shape_ptr)[kShape];
  // x must be rank 2 or 4
  if (x_shape.size() != 2 && x_shape.size() != 4) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", input x rank should be either 2 or 4, however "
                             << "input x shape is " << x_shape << ".";
  }
  // scale must be rank 1
  const int64_t input_num1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("scale rank", SizeToLong(scale_shape.size()), kEqual, input_num1, prim_name);
  // scale first dimension must be equal to x second dimension
  (void)CheckAndConvertUtils::CheckInteger("scale_shape shape[0]", SizeToLong(scale_shape[0]), kEqual,
                                           SizeToLong(x_shape[1]), prim_name);
  // Shape of scale、bias、mean and variance must be same
  std::map<std::string, ShapeVector> same_shape_args_map;
  (void)same_shape_args_map.insert(std::make_pair("shape of bias ", bias_shape));
  (void)same_shape_args_map.insert(std::make_pair("shape of mean ", mean_shape));
  (void)same_shape_args_map.insert(std::make_pair("shape of variance ", variance_shape));
  for (auto &elem : same_shape_args_map) {
    CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, scale_shape, prim_name);
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    x_shape_ptr, scale_shape_ptr, bias_shape_ptr, mean_shape_ptr, variance_shape_ptr});
}
}  // namespace

MIND_API_OPERATOR_IMPL(SyncBatchNorm, BaseOperator);
AbstractBasePtr SyncBatchNormInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  return abstract::MakeAbstract(SyncBatchNormInferShape(primitive, input_args),
                                SyncBatchNormInferType(primitive, input_args));
}

void SyncBatchNorm::Init(const float epsilon, const float momentum, const std::string group, const int64_t device_num) {
  set_epsilon(epsilon);
  set_momentum(momentum);
  set_group(group);
  set_device_num(device_num);
}

void SyncBatchNorm::set_epsilon(const float epsilon) {
  CheckAndConvertUtils::CheckInRange<float>(kEpsilon, epsilon, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kEpsilon, api::MakeValue(epsilon));
}

void SyncBatchNorm::set_momentum(const float momentum) {
  CheckAndConvertUtils::CheckInRange<float>(kMomentum, momentum, kIncludeBoth, {0.0, 1.0}, this->name());
  (void)this->AddAttr(kMomentum, api::MakeValue(momentum));
}

void SyncBatchNorm::set_group(const std::string group) { (void)this->AddAttr(kGroup, api::MakeValue(group)); }

void SyncBatchNorm::set_device_num(const int64_t device_num) {
  (void)this->AddAttr(kDeviceNum, api::MakeValue(device_num));
}

// AG means auto generated
class MIND_API AGSyncBatchNormInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SyncBatchNormInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SyncBatchNormInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SyncBatchNormInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SyncBatchNorm, prim::kPrimSyncBatchNorm, AGSyncBatchNormInfer, false);
}  // namespace ops
}  // namespace mindspore
