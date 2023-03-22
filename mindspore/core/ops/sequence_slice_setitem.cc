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

#include <string>
#include <vector>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

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
  for (auto arg : input_args) {
    MS_EXCEPTION_IF_NULL(arg);
  }
  auto sequence_abs = dyn_cast<abstract::AbstractSequence>(input_args[sequence_index]);
  MS_EXCEPTION_IF_NULL(sequence_abs);
  auto target_abs = dyn_cast<abstract::AbstractSequence>(input_args[target_index]);
  if (target_abs == nullptr) {
    MS_EXCEPTION(TypeError) << "Can only assign an iterable.";
  }
  auto start_abs = input_args[start_index];
  auto stop_abs = input_args[stop_index];
  auto step_abs = input_args[step_index];
  if (!sequence_abs->dynamic_len() && !target_abs->dynamic_len() && start_abs->BuildValue() != kAnyValue &&
      stop_abs->BuildValue() != kAnyValue && step_abs->BuildValue() != kAnyValue) {
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
    return SequenceSliceInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceSliceInferInner(prim, input_args)->BuildType();
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
