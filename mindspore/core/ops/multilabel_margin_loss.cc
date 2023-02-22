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

#include "ops/multilabel_margin_loss.h"

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
#include "ir/dtype/container.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/types.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::TupleShapePtr MultilabelMarginLossInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto x = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto target = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  const size_t xsizemin = 1;
  const size_t xsizemax = 2;
  if ((x.size() != xsizemin && x.size() != xsizemax) || (target.size() != xsizemin && target.size() != xsizemax)) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", the rank of input x and target should be 1 or 2, "
                             << "while rank of x is : " << x.size() << ", rank of target is : " << target.size() << ".";
  }
  if (x != target) {
    MS_EXCEPTION(ValueError) << "For " << op_name << ", x_shape and target_shape should be the same, "
                             << "while x_shape is : " << x << ", target_shape is : " << target << ".";
  }
  int64_t batch = x[kInputIndex0];
  ShapeVector out_shape0 = {batch};
  ShapeVector out_shape1 = target;
  int64_t reduction;
  CheckAndConvertUtils::GetReductionEnumValue(primitive->GetAttr(kReduction), &reduction);
  mindspore::Reduction reduction_ = static_cast<mindspore::Reduction>(reduction);
  if (reduction_ == REDUCTION_SUM || reduction_ == MEAN) {
    out_shape0.resize(0);
  }
  if (x.size() == xsizemin) {
    out_shape0.resize(0);
  }
  abstract::ShapePtr y_shape = std::make_shared<abstract::Shape>(out_shape0);
  abstract::ShapePtr istarget_shape = std::make_shared<abstract::Shape>(out_shape1);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_shape, istarget_shape});
}

TuplePtr MultilabelMarginLossInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types1 = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> valid_types2 = {kInt32};
  auto x = input_args[kInputIndex0]->BuildType();
  auto target = input_args[kInputIndex1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x, valid_types1, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("target", target, valid_types2, op_name);
  return std::make_shared<Tuple>(
    std::vector<TypePtr>{input_args[kInputIndex0]->BuildType(), input_args[kInputIndex1]->BuildType()});
}
}  // namespace

MIND_API_OPERATOR_IMPL(MultilabelMarginLoss, BaseOperator);
AbstractBasePtr MultilabelMarginLossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputsNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputsNum, primitive->name());
  auto infer_type = MultilabelMarginLossInferType(primitive, input_args);
  auto infer_shape = MultilabelMarginLossInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

int64_t MultilabelMarginLoss::get_reduction() const {
  static std::map<std::string, int64_t> kReductionModeMap{{"none", 0}, {"mean", 1}, {"sum", 2}};
  string reduc_str = GetValue<string>(GetAttr(kReduction));
  int64_t res = kReductionModeMap[reduc_str];
  return res;
}

// AG means auto generated
class MIND_API AGMultilabelMarginLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return MultilabelMarginLossInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return MultilabelMarginLossInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return MultilabelMarginLossInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(MultilabelMarginLoss, prim::kPrimMultilabelMarginLoss, AGMultilabelMarginLossInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
