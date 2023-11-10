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
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/nn_ops.h"
#include "ops/nth_element.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NthElementInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 0, kObjectTypeTensorType);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  (void)CheckAndConvertUtils::CheckInteger("input shape", SizeToLong(input_shape.size()), kGreaterEqual, 1,
                                           primitive->name());
  auto n_val = 0;
  if (CheckAndConvertUtils::IsTensor(input_args[1])) {
    const std::set<TypePtr> valid_types = {kInt32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("n", input_args[1]->GetType(), valid_types, primitive->name());
    auto n_value_ptr = input_args[1]->GetValue();
    auto n_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("n shape", SizeToLong(n_shape.size()), kEqual, 0, primitive->name());
    MS_EXCEPTION_IF_NULL(n_value_ptr);
    if (!n_value_ptr->ContainsValueAny()) {
      auto n_value_opt = GetArrayValue<int64_t>(n_value_ptr);
      if (!n_value_opt.has_value()) {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "' the n_value must be valid";
      }
      n_val = n_value_opt.value()[0];
    }
  } else if (CheckAndConvertUtils::IsScalar(input_args[1])) {
    auto n_value_ptr = input_args[1]->GetValue();
    if (!n_value_ptr->ContainsValueAny()) {
      auto n_value_opt = GetScalarValue<int64_t>(n_value_ptr);
      if (!n_value_opt.has_value()) {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "' the n_value must be valid";
      }
      n_val = n_value_opt.value();
    }
  } else {
    MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the n must be "
                            << "int or a scalar Tensor, but got " << input_args[1]->type_name() << ".";
  }

  (void)CheckAndConvertUtils::CheckInteger("n_value", n_val, kGreaterEqual, 0, primitive->name());
  if (input_shape.back() > 0) {
    (void)CheckAndConvertUtils::CheckInteger("n_value", n_val, kLessThan, input_shape.back(), primitive->name());
  }
  ShapeVector out_shape;
  int64_t len = SizeToLong(input_shape.size());
  for (int64_t i = 0; i < len - 1; i++) {
    (void)out_shape.emplace_back(input_shape[LongToSize(i)]);
  }
  auto return_shape = out_shape;
  return std::make_shared<abstract::Shape>(return_shape);
}
TypePtr NthElementInferType(const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt64, kInt32, kInt16, kInt8, kUInt8, kUInt16, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->GetType(), valid_types, primitive->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(NthElement, BaseOperator);

void NthElement::Init(const bool reverse) { this->set_reverse(reverse); }

void NthElement::set_reverse(const bool reverse) { (void)this->AddAttr(kReverse, api::MakeValue(reverse)); }

bool NthElement::get_reverse() const {
  auto value_ptr = GetAttr(kReverse);
  return GetValue<bool>(value_ptr);
}

abstract::AbstractBasePtr NthElementInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args) {
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, input_num, primitive->name());
  auto infer_type = NthElementInferType(primitive, input_args);
  auto infer_shape = NthElementInferShape(primitive, input_args)->shape();
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape);
}

// AG means auto generated
class MIND_API AGNthElementInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return NthElementInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return NthElementInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return NthElementInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(NthElement, prim::kPrimNthElement, AGNthElementInfer, false);
}  // namespace ops
}  // namespace mindspore
