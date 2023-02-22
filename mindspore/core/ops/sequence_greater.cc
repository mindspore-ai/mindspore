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

#include "ops/op_utils.h"
#include "ops/tuple_greater.h"
#include "ops/list_greater.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SequenceIsGreater(const AbstractBasePtrList &seqx_elements, const AbstractBasePtrList &seqy_elements,
                                  const std::string &prim_name, const bool &include_equal) {
  size_t x_len = seqx_elements.size();
  size_t y_len = seqy_elements.size();
  size_t max_len = std::max(x_len, y_len);

  for (size_t i = 0; i < max_len; ++i) {
    if (i >= x_len) {
      return std::make_shared<abstract::AbstractScalar>(false);
    }
    if (i >= y_len) {
      return std::make_shared<abstract::AbstractScalar>(true);
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
      return std::make_shared<abstract::AbstractScalar>(true);
    } else if (x < y) {
      return std::make_shared<abstract::AbstractScalar>(false);
    }
  }
  return std::make_shared<abstract::AbstractScalar>(include_equal);
}

AbstractBasePtr SequenceGreaterInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                          const bool &include_equal) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  constexpr size_t x_index = 0;
  constexpr size_t y_index = 1;
  auto x_abs = input_args[x_index];
  auto y_abs = input_args[y_index];
  if (!x_abs->isa<abstract::AbstractSequence>() && y_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For primitive '" << prim_name << "', the input must be a list or tuple, "
                            << "but got: " << x_abs->ToString() << " and " << y_abs->ToString();
  }
  auto seqx_abs = x_abs->cast<abstract::AbstractSequencePtr>();
  auto seqy_abs = y_abs->cast<abstract::AbstractSequencePtr>();
  if (seqx_abs->dynamic_len() || seqy_abs->dynamic_len()) {
    return std::make_shared<abstract::AbstractScalar>(kAnyValue, kBool);
  }
  const auto &seqx_elements = seqx_abs->elements();
  const auto &seqy_elements = seqx_abs->elements();
  return SequenceIsGreater(seqx_elements, seqy_elements, prim_name, include_equal);
}

class SequenceGreaterThanInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceGreaterInferInner(primitive, input_args, false)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceGreaterInferInner(prim, input_args, false)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceGreaterInferInner(primitive, input_args, false);
  }
};

class SequenceGreaterEqualInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceGreaterInferInner(primitive, input_args, true)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceGreaterInferInner(prim, input_args, true)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceGreaterInferInner(primitive, input_args, true);
  }
};

MIND_API_OPERATOR_IMPL(tuple_greater_than, BaseOperator);
MIND_API_OPERATOR_IMPL(list_greater_than, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(tuple_greater_than, prim::kPrimTupleGreaterThan, SequenceGreaterThanInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(list_greater_than, prim::kPrimListGreaterThan, SequenceGreaterThanInfer, false);

MIND_API_OPERATOR_IMPL(tuple_greater_equal, BaseOperator);
MIND_API_OPERATOR_IMPL(list_greater_equal, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(tuple_greater_equal, prim::kPrimTupleGreaterEqual, SequenceGreaterEqualInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(list_greater_equal, prim::kPrimListGreaterEqual, SequenceGreaterEqualInfer, false);
}  // namespace ops
}  // namespace mindspore
