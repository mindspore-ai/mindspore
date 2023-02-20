/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/ctc_loss_v2_grad.h"

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <set>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/common/utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
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
int64_t CTCLossV2Grad::get_blank() const { return GetValue<int64_t>(GetAttr("blank")); }
std::string CTCLossV2Grad::get_reduction() const { return GetValue<std::string>(GetAttr("reduction")); }
bool CTCLossV2Grad::get_zero_infinity() const { return GetValue<bool>(GetAttr("zero_infinity")); }
namespace {
abstract::ShapePtr CTCLossV2GradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kLenLogProbs = 3;
  auto prim_name = primitive->name();
  auto log_probs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  if (IsDynamicRank(log_probs_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  (void)CheckAndConvertUtils::CheckValue("dim of log_probs", log_probs_shape.size(), kEqual, kLenLogProbs, prim_name);
  int64_t T = log_probs_shape[kIndex0];
  int64_t N = log_probs_shape[kIndex1];
  int64_t C = log_probs_shape[kIndex2];
  ShapeVector output_shape = {T, N, C};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr CTCLossV2GradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[1]);
  (void)types.emplace("grad_out", input_args[0]->BuildType());
  (void)types.emplace("log_probs", input_args[1]->BuildType());
  auto out_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, name);
  return out_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CTCLossV2Grad, BaseOperator);
AbstractBasePtr CTCLossV2GradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t kInputNum = 7;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_shape = CTCLossV2GradInferShape(primitive, input_args);
  auto infer_type = CTCLossV2GradInferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGCTCLossV2GradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossV2GradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossV2GradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return CTCLossV2GradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(CTCLossV2Grad, prim::kPrimCTCLossV2Grad, AGCTCLossV2GradInfer, false);
}  // namespace ops
}  // namespace mindspore
