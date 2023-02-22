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
#include <vector>
#include <set>
#include <string>
#include <memory>

#include "ops/nth_element.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
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
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NthElementInferShape(const PrimitivePtr &primitive,
                                        const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  // support dynamic rank
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  (void)CheckAndConvertUtils::CheckInteger("input shape", SizeToLong(input_shape.size()), kGreaterEqual, 1,
                                           primitive->name());
  auto n_val = 0;
  if (input_args[1]->isa<abstract::AbstractTensor>()) {
    const std::set<TypePtr> valid_types = {kInt32};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("n", input_args[1]->BuildType(), valid_types, primitive->name());
    auto n = input_args[1]->cast<abstract::AbstractTensorPtr>();
    auto n_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShapeTrack())[kShape];
    (void)CheckAndConvertUtils::CheckInteger("n shape", SizeToLong(n_shape.size()), kEqual, 0, primitive->name());
    MS_EXCEPTION_IF_NULL(n);
    auto n_value_ptr = n->BuildValue();
    if (n_value_ptr->isa<tensor::Tensor>()) {
      MS_EXCEPTION_IF_NULL(n_value_ptr);
      auto n_tensor = n_value_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(n_tensor);
      n_val = *static_cast<int64_t *>(n_tensor->data_c());
    }
  } else if (input_args[1]->isa<abstract::AbstractScalar>()) {
    auto n = input_args[1]->cast<abstract::AbstractScalarPtr>();
    auto n_value_ptr = n->BuildValue();
    if (!n_value_ptr->isa<AnyValue>()) {
      if (!n_value_ptr->isa<Int64Imm>()) {
        MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the n"
                                << " must be a int, but got " << n_value_ptr->ToString() << ".";
      }
      n_val = GetValue<int64_t>(n_value_ptr);
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
  return CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->BuildType(), valid_types,
                                                    primitive->name());
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
