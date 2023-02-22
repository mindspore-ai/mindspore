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
#include "ops/approximate_equal.h"

#include <set>
#include <map>
#include <string>

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
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ApproximateEqualInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
  auto x1 = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x1);
  auto x2 = input_args[1]->BuildShape();
  MS_EXCEPTION_IF_NULL(x2);
  auto shape_ptr_x1 = x1->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr_x1);
  auto shape_ptr_x2 = x2->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_ptr_x2);
  if (!shape_ptr_x1->IsDynamic() && !shape_ptr_x2->IsDynamic()) {
    if (shape_ptr_x1->shape() != shape_ptr_x2->shape()) {
      MS_EXCEPTION(ArgumentError) << "For '" << prim_name
                                  << "', arg 'x1' must have the same shape as 'x2'. But got 'x1' shape: "
                                  << shape_ptr_x1->ToString() << ", 'x2' shape: " << shape_ptr_x2->ToString() << ".";
    }
  }
  return shape_ptr_x1;
}

TypePtr ApproximateEqualInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x1_dtype = input_args[0]->BuildType();
  auto x2_dtype = input_args[1]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_dtype, valid_types, prim->name());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", x2_dtype, valid_types, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x1", x1_dtype);
  (void)types.emplace("x2", x2_dtype);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  auto y_dtype = kBool;
  return y_dtype;
}
}  // namespace

void ApproximateEqual::Init(const float tolerance) { set_tolerance(tolerance); }

void ApproximateEqual::set_tolerance(const float tolerance) {
  (void)this->AddAttr(kTolerance, api::MakeValue(tolerance));
}

float ApproximateEqual::get_tolerance() const { return GetValue<float>(GetAttr(kTolerance)); }

MIND_API_OPERATOR_IMPL(ApproximateEqual, BaseOperator);
AbstractBasePtr ApproximateEqualInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = ApproximateEqualInferType(primitive, input_args);
  auto infer_shape = ApproximateEqualInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

// AG means auto generated
class MIND_API AGApproximateEqualInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ApproximateEqualInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ApproximateEqualInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ApproximateEqualInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(ApproximateEqual, prim::kPrimApproximateEqual, AGApproximateEqualInfer, false);
}  // namespace ops
}  // namespace mindspore
