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

#include "ops/sequence_slice_setitem.h"

#include <memory>
#include <string>
#include <vector>

#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr SequenceSliceInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_num = 5;
  constexpr size_t sequence_index = 0;
  constexpr size_t target_index = 1;
  constexpr size_t start_index = 2;
  constexpr size_t stop_index = 3;
  constexpr size_t step_index = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto sequence_abs = dyn_cast<abstract::AbstractSequence>(input_args[sequence_index]);
  MS_EXCEPTION_IF_NULL(sequence_abs);
  auto target_abs = dyn_cast<abstract::AbstractSequence>(input_args[target_index]);
  if (target_abs == nullptr) {
    MS_EXCEPTION(TypeError) << "Can only assign an iterable.";
  }
  auto start_abs = input_args[start_index];
  auto stop_abs = input_args[stop_index];
  auto step_abs = input_args[step_index];
  if (!sequence_abs->dynamic_len() && !target_abs->dynamic_len() && start_abs->GetValue() != kValueAny &&
      stop_abs->GetValue() != kValueAny && step_abs->GetValue() != kValueAny) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", the origin/target sequence should be dynamic length "
                             << "or one of start/stop/step should be variable.";
  }

  if (!sequence_abs->dynamic_len()) {
    sequence_abs = sequence_abs->Clone()->cast<abstract::AbstractSequencePtr>();
    sequence_abs->CheckAndConvertToDynamicLenSequence();
  }
  if (!target_abs->dynamic_len()) {
    target_abs = target_abs->Clone()->cast<abstract::AbstractSequencePtr>();
    target_abs->CheckAndConvertToDynamicLenSequence();
  }
  auto seq_element = sequence_abs->dynamic_len_element_abs();
  auto target_element = target_abs->dynamic_len_element_abs();
  auto ret = (sequence_abs == input_args[sequence_index]) ? sequence_abs->Clone()->cast<abstract::AbstractSequencePtr>()
                                                          : sequence_abs;
  if (target_element == nullptr) {
    return ret;
  }
  if (seq_element == nullptr) {
    ret->set_dynamic_len_element_abs(target_element);
    return ret;
  }
  const auto precondition_log = "For " + prim_name;
  const auto standard_abs_description = "element within origin sequence";
  const auto differ_abs_description = "element within target sequence";
  CheckAndConvertUtils::CheckAbstractTypeAndShapeSame(std::vector<AbstractBasePtr>{seq_element, target_element},
                                                      precondition_log, standard_abs_description,
                                                      differ_abs_description);
  return ret;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SequenceSliceSetItem, BaseOperator);
class SequenceSliceSetItemInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_num = 5;
    constexpr size_t sequence_index = 0;
    constexpr size_t target_index = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto sequence_shape = input_args[sequence_index]->GetShape()->cast<abstract::SequenceShapePtr>();
    MS_EXCEPTION_IF_NULL(sequence_shape);
    auto target_shape = input_args[target_index]->GetShape()->cast<abstract::SequenceShapePtr>();
    for (size_t i = 1; i < sequence_shape->size(); ++i) {
      if ((*sequence_shape)[i]->GetShapeVector() != (*sequence_shape)[i - 1]->GetShapeVector()) {
        MS_EXCEPTION(ValueError) << "SequenceShape[" << i - 1 << "]: " << (*sequence_shape)[i]->ToString()
                                 << " and SequenceShape[" << i << "]: " << (*sequence_shape)[i]->ToString()
                                 << " should be equal.";
      }
    }
    for (size_t i = 1; i < target_shape->size(); ++i) {
      if ((*target_shape)[i]->GetShapeVector() != (*target_shape)[i - 1]->GetShapeVector()) {
        MS_EXCEPTION(ValueError) << "TargetShape[" << i - 1 << "]: " << (*target_shape)[i]->ToString()
                                 << " and TargetShape[" << i << "]: " << (*target_shape)[i]->ToString()
                                 << " should be equal.";
      }
    }
    if (sequence_shape->size() == 0 && target_shape->size() == 0) {
      MS_EXCEPTION(ValueError) << "Sequence  and target cannot be all empty.";
    }
    if (sequence_shape->size() == 0) {
      return (*target_shape)[0]->Clone();
    }
    if (target_shape->size() == 0) {
      return (*sequence_shape)[0]->Clone();
    }
    return (*target_shape)[0]->Clone();
  }

  template <class T_PTR>
  TypePtr InferTypeInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_num = 5;
    constexpr size_t sequence_index = 0;
    constexpr size_t target_index = 1;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto sequence_type = input_args[sequence_index]->GetType()->cast<T_PTR>();
    MS_EXCEPTION_IF_NULL(sequence_type);
    auto target_type = input_args[target_index]->GetType()->cast<T_PTR>();
    for (size_t i = 1; i < sequence_type->size(); ++i) {
      if (!((*sequence_type)[i] == (*sequence_type)[i - 1])) {
        MS_EXCEPTION(ValueError) << "SequenceType[" << i - 1 << "]: " << (*sequence_type)[i]->ToString()
                                 << " and SequenceType[" << i << "]: " << (*sequence_type)[i]->ToString()
                                 << " should be equal.";
      }
    }
    for (size_t i = 1; i < target_type->size(); ++i) {
      if (!((*target_type)[i] == (*target_type)[i - 1])) {
        MS_EXCEPTION(ValueError) << "TargetType[" << i - 1 << "]: " << (*target_type)[i]->ToString()
                                 << " and TargetType[" << i << "]: " << (*target_type)[i]->ToString()
                                 << " should be equal.";
      }
    }
    if (sequence_type->size() == 0 && target_type->size() == 0) {
      MS_EXCEPTION(ValueError) << "Sequence  and target cannot be all empty.";
    }
    if (sequence_type->size() == 0) {
      return (*target_type)[0]->Clone();
    }
    if (target_type->size() == 0) {
      return (*sequence_type)[0]->Clone();
    }
    return (*target_type)[0]->Clone();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    constexpr size_t input_num = 5;
    constexpr size_t sequence_index = 0;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    if (CheckAndConvertUtils::IsTuple(input_args[sequence_index])) {
      return InferTypeInner<TuplePtr>(primitive, input_args);
    }
    if (CheckAndConvertUtils::IsList(input_args[sequence_index])) {
      return InferTypeInner<ListPtr>(primitive, input_args);
    }
    MS_EXCEPTION(TypeError) << "Unexpected sequence type: " << input_args[sequence_index]->ToString();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceSliceInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceSliceSetItem, prim::kPrimSequenceSliceSetItem, SequenceSliceSetItemInfer,
                                 false);
}  // namespace ops
}  // namespace mindspore
