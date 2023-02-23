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

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
template <typename T, typename G>
AbstractBasePtr FindMaxOrMin(const AbstractBasePtrList &seq_elements, const bool is_max) {
  std::vector<T> values;
  for (size_t i = 0; i < seq_elements.size(); ++i) {
    auto element = seq_elements[i];
    if (element->BuildValue() == kAnyValue) {
      return element->Clone();
    }
    values.push_back(element->BuildValue()->cast<G>()->value());
  }
  if (is_max) {
    return std::make_shared<abstract::AbstractScalar>(*std::max_element(values.begin(), values.end()));
  }
  return std::make_shared<abstract::AbstractScalar>(*std::min_element(values.begin(), values.end()));
}

AbstractBasePtr SequenceMaxMinInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                         bool is_max = true) {
  std::string op_name = primitive->name();
  constexpr size_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto arg = input_args[0];
  auto seq_abs = arg->cast<abstract::AbstractSequencePtr>();
  if (seq_abs->dynamic_len()) {
    auto seq_type = seq_abs->BuildType();
    auto type = seq_type->cast<TuplePtr>()->dynamic_element_type();
    return std::make_shared<abstract::AbstractScalar>(kAnyValue, type == nullptr ? kAnyType : type);
  }
  const auto &seq_elements = seq_abs->elements();
  auto type = seq_abs->elements()[0]->BuildType();
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

MIND_API_OPERATOR_IMPL(SequenceMax, BaseOperator);
class SequenceMaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferInner(prim, input_args)->BuildType();
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
    return SequenceMaxMinInferInner(primitive, input_args, false)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMaxMinInferInner(prim, input_args, false)->BuildType();
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
