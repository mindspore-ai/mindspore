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

#include "ops/grad/einsum_grad.h"

#include <memory>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shared_ptr.h"
#include "mindapi/ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputNum2 = 2;
abstract::BaseShapePtr EinsumGradInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  abstract::BaseShapePtrList elements;
  if (input_args.size() == kInputNum2 && CheckAndConvertUtils::IsSequence(input_args[kIndex0])) {
    auto seq_shape = input_args[kIndex0]->GetShape()->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(seq_shape);
    elements = seq_shape->shape();
  } else {
    for (int64_t idx = 0; idx < SizeToLong(input_args.size()) - 1; idx++) {
      if (CheckAndConvertUtils::IsSequence(input_args[idx])) {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
      }
      elements.emplace_back(input_args[idx]->GetShape());
    }
  }
  return std::make_shared<abstract::TupleShape>(elements);
}

TypePtr EinsumGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  bool is_tuple = true;
  TypePtrList elements;
  if (input_args.size() == kInputNum2 && CheckAndConvertUtils::IsSequence(input_args[kIndex0])) {
    if (CheckAndConvertUtils::IsTuple(input_args[kIndex0])) {
      auto seq_type = input_args[kIndex0]->GetType()->cast<TuplePtr>();
      MS_EXCEPTION_IF_NULL(seq_type);
      elements = seq_type->elements();
    } else {
      auto seq_type = input_args[kIndex0]->GetType()->cast<ListPtr>();
      MS_EXCEPTION_IF_NULL(seq_type);
      elements = seq_type->elements();
      is_tuple = false;
    }
  } else {
    for (int64_t idx = 0; idx < SizeToLong(input_args.size()) - 1; idx++) {
      if (CheckAndConvertUtils::IsSequence(input_args[idx])) {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the input data type must be list or tuple of tensors.";
      }
      elements.emplace_back(input_args[idx]->GetType());
    }
  }
  auto out_type =
    is_tuple ? std::make_shared<Tuple>(elements)->cast<TypePtr>() : std::make_shared<List>(elements)->cast<TypePtr>();
  return out_type;
}
}  // namespace
MIND_API_OPERATOR_IMPL(EinsumGrad, BaseOperator);
void EinsumGrad::Init(const std::string equation) { this->set_equation(equation); }

void EinsumGrad::set_equation(const std::string equation) { (void)this->AddAttr(kEquation, api::MakeValue(equation)); }

std::string EinsumGrad::get_equation() const {
  auto value_ptr = this->GetAttr(kEquation);
  return GetValue<std::string>(value_ptr);
}

abstract::AbstractBasePtr EinsumGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args) {
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kInputNum2,
                                           primitive->name());
  return abstract::MakeAbstract(EinsumGradInferShape(primitive, input_args),
                                EinsumGradInferType(primitive, input_args));
}

// AG means auto generated
class MIND_API AGEinsumGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return EinsumGradInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return EinsumGradInferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return EinsumGradInfer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(EinsumGrad, prim::kPrimEinsumGrad, AGEinsumGradInfer, false);
}  // namespace ops
}  // namespace mindspore
