/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/lars_v2_update.h"

#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LARSUpdateInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  MS_LOG(INFO) << "For '" << op_name << "', it's now doing infer shape.";
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack());
  auto gradient_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack());
  auto norm_weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack());
  auto norm_gradient_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->GetShapeTrack());
  auto weight_decay_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack());
  auto learning_rate_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShapeTrack());

  if (weight_shape[kShape].size() != gradient_shape[kShape].size()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', weight shape size must be equal to gradient shape size, but got "
                             << "weight shape size: " << weight_shape[kShape].size()
                             << ", gradient shape size: " << gradient_shape[kShape].size() << ".";
  }
  if (norm_weight_shape[kShape].size() != norm_gradient_shape[kShape].size()) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << "', norm weight shape size must be equal to norm gradient shape size, but got "
                             << "norm weight shape size: " << norm_weight_shape[kShape].size()
                             << ", norm gradient shape size: " << norm_gradient_shape[kShape].size() << ".";
  }
  for (size_t index = 0; index < weight_shape[kShape].size(); index++) {
    if (weight_shape[kShape][index] != gradient_shape[kShape][index]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << index
                               << "th dim of weight shape and gradient shape must be equal, but got "
                               << "weight shape[" << index << "]: " << weight_shape[kShape][index]
                               << ", gradient shape[" << index << "]: " << gradient_shape[kShape][index] << ".";
    }
  }
  for (size_t index = 0; index < norm_weight_shape[kShape].size(); index++) {
    if (norm_weight_shape[kShape][index] != norm_gradient_shape[kShape][index]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << index
                               << "th dim of norm weight shape and norm gradient shape must be equal, but got "
                               << "norm weight shape[" << index << "]: " << norm_weight_shape[kShape][index]
                               << ", norm gradient shape[" << index << "]: " << norm_gradient_shape[kShape][index]
                               << ".";
    }
  }
  auto shp_len = weight_decay_shape[kShape].size();
  auto para_name = input_args[4]->ToString();
  (void)CheckAndConvertUtils::CheckInteger(para_name, SizeToLong(shp_len), kLessEqual, 1);
  if (shp_len == 1) {
    (void)CheckAndConvertUtils::CheckInteger(para_name, weight_decay_shape[kShape][0], kEqual, 1);
  }
  shp_len = learning_rate_shape[kShape].size();
  para_name = input_args[5]->ToString();
  (void)CheckAndConvertUtils::CheckInteger(para_name, SizeToLong(shp_len), kLessEqual, 1);
  if (shp_len == 1) {
    (void)CheckAndConvertUtils::CheckInteger(para_name, learning_rate_shape[kShape][0], kEqual, 1);
  }

  return std::make_shared<abstract::Shape>(weight_shape[kShape]);
}

TypePtr LARSUpdateInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 6;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("Weight dtype", input_args[0]->BuildType());
  (void)types.emplace("gradient dtype", input_args[1]->BuildType());
  (void)types.emplace("norm weight dtype", input_args[2]->BuildType());
  (void)types.emplace("norm gradient dtype", input_args[3]->BuildType());
  const std::set<TypePtr> valid_types = {kInt16, kInt32, kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(types, valid_types, primitive->name(), true);
  return types["Weight dtype"];
}
}  // namespace

MIND_API_OPERATOR_IMPL(LARSUpdate, BaseOperator);
AbstractBasePtr LARSUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = LARSUpdateInferType(primitive, input_args);
  auto infer_shape = LARSUpdateInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGLARSUpdateInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LARSUpdateInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LARSUpdateInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return LARSUpdateInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(LARSUpdate, prim::kPrimLARSUpdate, AGLARSUpdateInfer, false);
}  // namespace ops
}  // namespace mindspore
