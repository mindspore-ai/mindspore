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

#include "ops/grad/multilabel_margin_loss_grad.h"

#include <algorithm>
#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MultilabelMarginLossGradInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto target = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  if (IsDynamic(x) || IsDynamic(target)) {
    return std::make_shared<abstract::Shape>(x);
  }
  if ((x.size() != kDim1 && x.size() != kDim2) || (target.size() != kDim1 && target.size() != kDim2)) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", the rank of input x and target should be 1 or 2, "
                             << "while rank of x is : " << x.size() << ", rank of target is : " << target.size() << ".";
  }
  if (x != target) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", x_shape and target_shape should be the same, "
                             << "while x_shape is : " << x << ", target_shape is : " << target << ".";
  }
  return std::make_shared<abstract::Shape>(x);
}

TypePtr MultilabelMarginLossGradInferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types1 = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_types2 = {kInt32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("y_grad", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("x", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types1, op_name);
  auto target = input_args[kInputIndex2]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("target", target, valid_types2, op_name);
  return input_args[kInputIndex1]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(MultilabelMarginLossGrad, BaseOperator);
AbstractBasePtr MultilabelMarginLossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = MultilabelMarginLossGradInferType(primitive, input_args);
  auto infer_shape = MultilabelMarginLossGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

int64_t MultilabelMarginLossGrad::get_reduction() const {
  static std::map<std::string, int64_t> kReductionModeMap{{"none", 0}, {"mean", 1}, {"sum", 2}};
  string reduc_str = GetValue<string>(GetAttr(kReduction));
  int64_t res = kReductionModeMap[reduc_str];
  return res;
}

// AG means auto generated
class MIND_API AGMultilabelMarginLossGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MultilabelMarginLossGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MultilabelMarginLossGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MultilabelMarginLossGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MultilabelMarginLossGrad, prim::kPrimMultilabelMarginLossGrad,
                                 AGMultilabelMarginLossGradInfer, false);
}  // namespace ops
}  // namespace mindspore
