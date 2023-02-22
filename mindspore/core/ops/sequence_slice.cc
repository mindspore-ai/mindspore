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

#include "ops/sequence_slice.h"

#include <vector>
#include <string>
#include <set>
#include <memory>

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
int64_t SequenceSliceGetValue(const std::string &prim_name, const std::string &attr_name, const AbstractBasePtr &abs) {
  auto build_type = abs->BuildType();
  auto build_value = abs->BuildValue();
  if (build_type == kInt32) {
    return GetValue<int32_t>(build_value);
  } else if (build_type == kInt64) {
    return GetValue<int64_t>(build_value);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the type of '" << attr_name
                            << "' should be int32, int64 but got: " << abs->BuildType()->ToString();
  }
}
AbstractBasePtr SliceInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_num = 4;
  constexpr size_t seq_index = 0;
  constexpr size_t start_index = 1;
  constexpr size_t end_index = 2;
  constexpr size_t step_index = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto first_abs = input_args[seq_index];
  MS_EXCEPTION_IF_NULL(first_abs);
  if (!first_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the first input should be tuple or list but got: " << first_abs->ToString();
  }
  auto seq_abs = first_abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abs);
  if (seq_abs->dynamic_len()) {
    // If the length of input sequence is dynamic length, the length of sliced sequence should also be dynamic length.
    return seq_abs->Clone();
  }
  auto start_abs = input_args[start_index];
  MS_EXCEPTION_IF_NULL(start_abs);
  auto end_abs = input_args[end_index];
  MS_EXCEPTION_IF_NULL(end_abs);
  auto step_abs = input_args[step_index];
  MS_EXCEPTION_IF_NULL(step_abs);

  // all value is known
  if (start_abs->BuildValue() != kAnyValue && end_abs->BuildValue() != kAnyValue &&
      step_abs->BuildValue() != kAnyValue) {
    int64_t start_v, end_v, step_v;
    const std::string start_str = "start";
    const std::string end_str = "end";
    const std::string step_str = "step";
    start_v = SequenceSliceGetValue(prim_name, start_str, start_abs);
    end_v = SequenceSliceGetValue(prim_name, end_str, end_abs);
    step_v = SequenceSliceGetValue(prim_name, step_str, step_abs);
    int64_t len = seq_abs->elements().size();
    auto output_size = SequenceSliceGetOutputSize(start_v, end_v, step_v, len);
    abstract::AbstractBasePtrList abs{};
    for (int64_t i = 0; i < output_size; i++) {
      abs.push_back(std::make_shared<abstract::AbstractScalar>(kAnyValue, seq_abs->ElementsType()[0]));
    }
    auto ret = std::make_shared<abstract::AbstractTuple>(abs);
    return ret;
  }
  auto ret = seq_abs->Clone()->cast<abstract::AbstractSequencePtr>();
  ret->CheckAndConvertToDynamicLenSequence();
  return ret;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SequenceSlice, BaseOperator);
class SequenceSliceInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SliceInferInner(primitive, input_args);
  }
  std::set<int64_t> GetValueDependArgIndices() const override { return {1, 2, 3}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceSlice, prim::kPrimSequenceSlice, SequenceSliceInfer, false);
}  // namespace ops
}  // namespace mindspore
