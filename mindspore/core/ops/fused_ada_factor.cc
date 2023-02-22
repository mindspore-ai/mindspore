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

#include "ops/fused_ada_factor.h"

#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kParamIndex = 7;
constexpr size_t kFusedAdaFactorInputsNum = 12;
auto constexpr kEnableScaleParameter = "enable_scale_parameter";
auto constexpr kEnableFirstMoment = "enable_first_moment";
auto constexpr kEnableWeightDecay = "enable_weight_decay";
abstract::TupleShapePtr FusedAdaFactorInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto param_shape_r = input_args[kParamIndex]->Broaden()->BuildShape();
  auto outputs = std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>({param_shape_r}));
  return outputs;
}

TypePtr FusedAdaFactorInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto type = input_args[kParamIndex]->BuildType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{type});
}
}  // namespace

void FusedAdaFactor::set_enable_scale_parameter(bool flag) {
  (void)this->AddAttr(kEnableScaleParameter, api::MakeValue(flag));
}

bool FusedAdaFactor::get_enable_scale_parameter() const {
  auto value_ptr = GetAttr(kEnableScaleParameter);
  return GetValue<bool>(value_ptr);
}

void FusedAdaFactor::set_enable_first_moment(bool flag) {
  (void)this->AddAttr(kEnableFirstMoment, api::MakeValue(flag));
}

bool FusedAdaFactor::get_enable_first_moment() const {
  auto value_ptr = GetAttr(kEnableFirstMoment);
  return GetValue<bool>(value_ptr);
}

void FusedAdaFactor::set_enable_weight_decay(bool flag) {
  (void)this->AddAttr(kEnableWeightDecay, api::MakeValue(flag));
}

bool FusedAdaFactor::get_enable_weight_decay() const {
  auto value_ptr = GetAttr(kEnableWeightDecay);
  return GetValue<bool>(value_ptr);
}

MIND_API_OPERATOR_IMPL(FusedAdaFactor, BaseOperator);
MIND_API_OPERATOR_IMPL(FusedAdaFactorWithGlobalNorm, FusedAdaFactor);
AbstractBasePtr FusedAdaFactorInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual,
                                           SizeToLong(kFusedAdaFactorInputsNum), op_name);
  auto types = FusedAdaFactorInferType(primitive, input_args);
  auto shapes = FusedAdaFactorInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

// AG means auto generated
class MIND_API AGFusedAdaFactorInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FusedAdaFactorInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FusedAdaFactorInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return FusedAdaFactorInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(FusedAdaFactor, prim::kPrimFusedAdaFactor, AGFusedAdaFactorInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(FusedAdaFactorWithGlobalNorm, prim::kPrimFusedAdaFactorWithGlobalNorm,
                                 AGFusedAdaFactorInfer, false);
}  // namespace ops
}  // namespace mindspore
