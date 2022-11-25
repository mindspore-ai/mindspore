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

#include "ops/sequence_count.h"

#include <vector>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SequenceCountInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractScalar>(kAnyValue, kInt64);
}

bool ComparesTwoValues(const ValuePtr &value_1, const ValuePtr &value_2) {
  MS_EXCEPTION_IF_NULL(value_1);
  MS_EXCEPTION_IF_NULL(value_2);
  if (!value_1->IsSameTypeId(value_2->tid())) {
    return false;
  }
  if (value_1->isa<tensor::Tensor>()) {
    auto list_tensor_value = value_2->cast_ptr<tensor::Tensor>();
    MS_EXCEPTION_IF_NULL(list_tensor_value);
    return value_1->cast_ptr<tensor::Tensor>()->ValueEqual(*list_tensor_value);
  }
  return *value_1 == *value_2;
}

ValuePtr SequenceCountInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const size_t input_num = 2;
  auto prim_name = prim->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  constexpr size_t seq_index = 0;
  constexpr size_t target_index = 1;
  auto input_abs = input_args[seq_index];
  auto target_abs = input_args[target_index];
  if (!input_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For primitive '" << prim_name << "', the first input must be a list or tuple, "
                            << "but got: " << input_abs->ToString();
  }
  auto seq_abs = input_abs->cast<abstract::AbstractSequencePtr>();
  if (seq_abs->dynamic_len()) {
    return nullptr;
  }
  auto target_value = target_abs->BuildValue();
  if (seq_abs->BuildValue() == kAnyValue || target_value == kAnyValue) {
    return nullptr;
  }
  const auto &seq_elements = seq_abs->elements();
  int64_t count = 0;
  for (auto element : seq_elements) {
    if (ComparesTwoValues(target_value, element->BuildValue())) {
      ++count;
    }
  }
  return MakeValue(count);
}

MIND_API_OPERATOR_IMPL(SequenceCount, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SequenceCount, prim::kPrimSequenceCount, SequenceCountInfer, SequenceCountInferValue,
                             true);
}  // namespace ops
}  // namespace mindspore
