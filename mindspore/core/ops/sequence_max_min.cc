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

#include "ops/sequence_max_min.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
template <typename T, typename G>
AbstractBasePtr FindMaxOrMin(const AbstractBasePtrList &seq_elements, const bool is_max) {
  std::vector<T> values;
  for (size_t i = 0; i < seq_elements.size(); ++i) {
    auto element = seq_elements[i];
    if (element->GetValue()->ContainsValueAny()) {
      return element->Clone();
    }
    values.push_back(element->GetValue()->cast<G>()->value());
  }
  if (is_max) {
    return std::make_shared<abstract::AbstractScalar>(*std::max_element(values.begin(), values.end()));
  }
  return std::make_shared<abstract::AbstractScalar>(*std::min_element(values.begin(), values.end()));
}

AbstractBasePtr SequenceMaxMinInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                         bool is_max = true) {
  const auto &op_name = primitive->name();
  constexpr size_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, SizeToLong(input_num), op_name);
  auto arg = input_args[0];
  auto seq_abs = arg->cast<abstract::AbstractSequencePtr>();
  if (seq_abs->dynamic_len()) {
    auto seq_type = seq_abs->GetType();
    TypePtr type = nullptr;
    if (seq_type->isa<List>()) {
      type = seq_type->cast<ListPtr>()->dynamic_element_type();
    } else if (seq_type->isa<Tuple>()) {
      type = seq_type->cast<TuplePtr>()->dynamic_element_type();
    } else {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "' is not supported" << seq_type->ToString() << '.';
    }
    return std::make_shared<abstract::AbstractScalar>(kValueAny, type == nullptr ? kTypeAny : type);
  }
  const auto &seq_elements = seq_abs->elements();
  auto type = seq_elements[0]->GetType();
  if (type->type_id() == kInt64->type_id()) {
    return FindMaxOrMin<int64_t, Int64ImmPtr>(seq_elements, is_max);
  } else if (type->type_id() == kInt32->type_id()) {
    return FindMaxOrMin<int, Int32ImmPtr>(seq_elements, is_max);
  } else if (type->type_id() == kFloat32->type_id()) {
    return FindMaxOrMin<float, FP32ImmPtr>(seq_elements, is_max);
  } else if (type->type_id() == kFloat64->type_id()) {
    return FindMaxOrMin<double, FP64ImmPtr>(seq_elements, is_max);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << op_name << "' is not supported" << type->ToString() << '.';
  }
}

BaseShapePtr SequenceMaxMinInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto queue_shape = input_args[kIndex0]->GetShape()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(queue_shape);
  return queue_shape->shape()[kIndex0]->Clone();
}

TypePtr SequenceMaxMinInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  if (CheckAndConvertUtils::IsTuple(input_args[kIndex0])) {
    auto queue_type = input_args[kIndex0]->GetType()->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(queue_type);
    return queue_type->elements()[kIndex0]->Clone();
  } else {
    auto queue_type = input_args[kIndex0]->GetType()->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(queue_type);
    return queue_type->elements()[kIndex0]->Clone();
  }
}
}  // namespace

MIND_API_OPERATOR_IMPL(SequenceMax, BaseOperator);
class SequenceMaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferType(prim, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferInner(primitive, input_args);
  }
};

MIND_API_OPERATOR_IMPL(SequenceMin, BaseOperator);
class SequenceMinInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferType(prim, input_args);
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferInner(primitive, input_args, false);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceMin, prim::kPrimSequenceMin, SequenceMinInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceMax, prim::kPrimSequenceMax, SequenceMaxInfer, false);
}  // namespace ops
}  // namespace mindspore
