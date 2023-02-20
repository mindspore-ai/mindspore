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

#include "ops/grad/pdist_grad.h"

#include <map>
#include <memory>
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
#include "utils/overload.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PdistGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto pdist_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  auto x_size = x_shape.size();
  if (!IsDynamic(grad_shape) && !IsDynamic(pdist_shape)) {
    (void)CheckAndConvertUtils::CheckValue("y_grad shape", grad_shape, kEqual, "y shape", pdist_shape, prim_name);
  }
  const size_t x_dim = 2;
  if (!IsDynamicRank(x_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("x dim", x_size, kEqual, x_dim, "PdistGrad");
  }

  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr PdistGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const size_t y_grad_index = 0;
  const size_t x_index = 1;
  const size_t y_index = 2;
  const std::set<TypePtr> valid_types = {kFloat64, kFloat32, kFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("y_grad", input_args[y_grad_index]->BuildType());
  (void)types.emplace("x", input_args[x_index]->BuildType());
  (void)types.emplace("y", input_args[y_index]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
}
}  // namespace

float PdistGrad::get_p() const {
  auto value_ptr = this->GetAttr(kP);
  return GetValue<float>(value_ptr);
}
void PdistGrad::set_p(const float p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

MIND_API_OPERATOR_IMPL(PdistGrad, BaseOperator);
AbstractBasePtr PdistGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = PdistGradInferType(primitive, input_args);
  auto infer_shape = PdistGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGPdistGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return PdistGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return PdistGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return PdistGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PdistGrad, prim::kPrimPdistGrad, AGPdistGradInfer, false);
}  // namespace ops
}  // namespace mindspore
