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
#include "ops/grad/soft_margin_loss_grad.h"

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputSize = 3;
abstract::ShapePtr SoftMarginLossGradInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize, op_name);
  auto predict = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto label = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto dout = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("logits shape", predict, kEqual, "labels shape", label, op_name, ValueError);
  if (dout.size() > 1) {
    CheckAndConvertUtils::Check("logits shape", predict, kEqual, "dout shape", dout, op_name, ValueError);
  }
  return std::make_shared<abstract::Shape>(predict);
}

TypePtr SoftMarginLossGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputSize, op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("logits", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("labels", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("dout", input_args[kInputIndex2]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return CheckAndConvertUtils::CheckTensorTypeValid("logits", input_args[0]->BuildType(), valid_types, op_name);
}
}  // namespace

AbstractBasePtr SoftMarginLossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(SoftMarginLossGradInferShape(primitive, input_args),
                                SoftMarginLossGradInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(SoftMarginLossGrad, prim::kPrimSoftMarginLossGrad, SoftMarginLossGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
