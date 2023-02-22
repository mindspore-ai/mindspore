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
#include "ops/nan_to_num.h"

#include <set>
#include <string>
#include <vector>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/utils.h"
#include "base/float16.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/base/type_id.h"
#include "mindapi/ir/value.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NanToNumInferShape(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  auto x = input_args[kInputIndex0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr NanToNumInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::string op_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, kInputIndex0);
  auto x_dtype = input_args[kInputIndex0]->BuildType();
  MS_EXCEPTION_IF_NULL(x_dtype);
  const std::set<TypePtr> x_valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, x_valid_types, op_name);
  auto dtype = x_dtype->cast<TensorTypePtr>();
  TypeId dtype_id = dtype->element()->type_id();

  auto nan_none = prim->GetAttr("nan_none");
  if (nan_none != nullptr && GetValue<bool>(nan_none)) {
    prim->set_attr("nan", MakeValue(static_cast<float>(0.0)));
  }
  auto posinf_none = prim->GetAttr("posinf_none");
  if (posinf_none != nullptr && GetValue<bool>(posinf_none)) {
    if (dtype_id == kNumberTypeFloat32) {
      prim->set_attr("posinf", MakeValue(std::numeric_limits<float>::max()));
    } else if (dtype_id == kNumberTypeFloat16) {
      prim->set_attr("posinf", MakeValue(static_cast<float>(std::numeric_limits<float16>::max())));
    }
  }
  auto neginf_none = prim->GetAttr("neginf_none");
  if (neginf_none != nullptr && GetValue<bool>(neginf_none)) {
    if (dtype_id == kNumberTypeFloat32) {
      prim->set_attr("neginf", MakeValue(std::numeric_limits<float>::lowest()));
    } else if (dtype_id == kNumberTypeFloat16) {
      prim->set_attr("neginf", MakeValue(static_cast<float>(std::numeric_limits<float16>::lowest())));
    }
  }

  return x_dtype;
}
}  // namespace

AbstractBasePtr NanToNumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = NanToNumInferType(primitive, input_args);
  auto infer_shape = NanToNumInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

void NanToNum::Init(float nan, float posinf, float neginf) {
  set_nan_value(nan);
  set_posinf_value(posinf);
  set_neginf_value(neginf);
}

void NanToNum::set_nan_value(float nan_value) { (void)this->AddAttr(kNan, api::MakeValue(nan_value)); }

float NanToNum::get_nan_value() const { return GetValue<float>(GetAttr(kNan)); }

void NanToNum::set_posinf_value(float posinf_value) { (void)this->AddAttr(kPosinf, api::MakeValue(posinf_value)); }

float NanToNum::get_posinf_value() const { return GetValue<float>(GetAttr(kPosinf)); }

void NanToNum::set_neginf_value(float neginf_value) { (void)this->AddAttr(kNeginf, api::MakeValue(neginf_value)); }

float NanToNum::get_neginf_value() const { return GetValue<float>(GetAttr(kNeginf)); }

MIND_API_OPERATOR_IMPL(NanToNum, BaseOperator);

// AG means auto generated
class MIND_API AGNanToNumInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NanToNumInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NanToNumInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NanToNumInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NanToNum, prim::kPrimNanToNum, AGNanToNumInfer, false);
}  // namespace ops
}  // namespace mindspore
