/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include <memory>
#include <string>
#include <algorithm>

#include "ops/tuple_le.h"
#include "ops/tuple_lt.h"
#include "ops/list_le.h"
#include "ops/list_lt.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr LessImpl(const AbstractBasePtrList &seqx_elements, const AbstractBasePtrList &seqy_elements,
                         const std::string &prim_name, const bool is_less_equal = true) {
  size_t x_size = seqx_elements.size();
  size_t y_size = seqy_elements.size();
  size_t max_size = std::max(x_size, y_size);

  for (size_t i = 0; i < max_size; ++i) {
    if (i >= x_size) {
      return std::make_shared<abstract::AbstractScalar>(true);
    }
    if (i >= y_size) {
      return std::make_shared<abstract::AbstractScalar>(false);
    }
    auto x_element = seqx_elements[i];
    auto y_element = seqy_elements[i];

    if (x_element->BuildType()->type_id() == kObjectTypeTensorType ||
        y_element->BuildType()->type_id() == kObjectTypeTensorType) {
      MS_EXCEPTION(TypeError) << "For primitive tupel_equal, the input element must be scalar, but got "
                              << x_element->ToString() << " and " << y_element->ToString();
    }
    if (x_element->BuildValue() == kAnyValue || y_element->BuildValue() == kAnyValue) {
      return std::make_shared<abstract::AbstractScalar>(kAnyValue, kBool);
    }

    auto x = GetScalarValue<double>(prim_name, x_element->BuildValue());
    auto y = GetScalarValue<double>(prim_name, y_element->BuildValue());
    if (x > y) {
      return std::make_shared<abstract::AbstractScalar>(false);
    } else if (x < y) {
      return std::make_shared<abstract::AbstractScalar>(true);
    }
  }
  return std::make_shared<abstract::AbstractScalar>(is_less_equal);
}

AbstractBasePtr SequenceLessInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                       const bool is_less_equal = true) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_abs = input_args[0];
  auto y_abs = input_args[1];
  if (!x_abs->isa<abstract::AbstractSequence>() || !y_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For primitive '" << prim_name << "', the input must be a list or tuple, "
                            << "but got: " << x_abs->ToString() << " and " << y_abs->ToString();
  }
  auto seqx_abs = x_abs->cast<abstract::AbstractSequencePtr>();
  auto seqy_abs = y_abs->cast<abstract::AbstractSequencePtr>();
  if (seqx_abs->dynamic_len() || seqy_abs->dynamic_len()) {
    return std::make_shared<abstract::AbstractScalar>(kAnyValue, kBool);
  }
  const auto &seqx_elements = seqx_abs->elements();
  const auto &seqy_elements = seqy_abs->elements();

  return LessImpl(seqx_elements, seqy_elements, prim_name, is_less_equal);
}

class SequenceLessThanInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceLessInferInner(primitive, input_args, false)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceLessInferInner(prim, input_args, false)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceLessInferInner(primitive, input_args, false);
  }
};

class SequenceLessEqualInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceLessInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceLessInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceLessInferInner(primitive, input_args);
  }
};

MIND_API_OPERATOR_IMPL(tuple_le, BaseOperator);
MIND_API_OPERATOR_IMPL(tuple_lt, BaseOperator);
MIND_API_OPERATOR_IMPL(list_le, BaseOperator);
MIND_API_OPERATOR_IMPL(list_lt, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(tuple_le, prim::kPrimTupleLessEqual, SequenceLessEqualInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(list_le, prim::kPrimListLessEqual, SequenceLessEqualInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(tuple_lt, prim::kPrimTupleLessThan, SequenceLessThanInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(list_lt, prim::kPrimListLessThan, SequenceLessThanInfer, false);
}  // namespace ops
}  // namespace mindspore
