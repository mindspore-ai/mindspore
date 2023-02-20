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

#include "ops/grad/soft_shrink_grad.h"

#include <set>
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void SoftShrinkGrad::Init(const float &lambd) {
  set_lambd(lambd);
  return;
}

void SoftShrinkGrad::set_lambd(const float &lambd) {
  (void)AddAttr(kLambd, api::MakeValue(lambd));
  return;
}

float SoftShrinkGrad::get_lambd() const {
  auto value_ptr = GetAttr(kLambd);
  return GetValue<float>(value_ptr);
}

namespace {
abstract::ShapePtr SoftShrinkGradInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  auto input_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto input_x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto prim_name = primitive->name();
  CheckAndConvertUtils::Check("input_grad_shape", input_grad_shape, kEqual, input_x_shape, prim_name, TypeError);
  return std::make_shared<abstract::Shape>(input_grad_shape);
}

TypePtr SoftShrinkGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_grad", input_args[0]->BuildType());
  (void)types.emplace("input_x", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

AbstractBasePtr SoftShrinkGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = SoftShrinkGradInferType(primitive, input_args);
  auto infer_shape = SoftShrinkGradInferShape(primitive, input_args);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}

MIND_API_OPERATOR_IMPL(SoftShrinkGrad, BaseOperator);

// AG means auto generated
class MIND_API AGSoftShrinkGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftShrinkGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftShrinkGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SoftShrinkGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(SoftShrinkGrad, prim::kPrimSoftShrinkGrad, AGSoftShrinkGradInfer, false);
}  // namespace ops
}  // namespace mindspore
