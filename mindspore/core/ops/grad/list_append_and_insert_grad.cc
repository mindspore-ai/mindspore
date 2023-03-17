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

#include "ops/grad/list_append_and_insert_grad.h"

#include <vector>
#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr ListAppendAndInsertGradInnerInfer(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t data_index = 0;
  constexpr size_t index_index = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto data_abs = dyn_cast<abstract::AbstractSequence>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);
  auto index_abs = abstract::CheckArg<abstract::AbstractScalar>(prim_name, input_args, index_index);
  if (!data_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "The prim '" << prim_name
                            << "', the input_data must be list, index must be scalar, but got " << data_abs->ToString()
                            << " target is " << index_abs->ToString();
  }

  if (data_abs->dynamic_len()) {
    return data_abs->Clone();
  }

  const auto &elements = data_abs->elements();
  if (elements.empty()) {
    MS_EXCEPTION(ValueError) << "The prim '" << prim_name << "', pop from empty list";
  }
  abstract::AbstractBasePtrList abs;
  for (size_t i = 0; i < data_abs->size(); ++i) {
    abs.push_back(data_abs->elements()[i]);
  }
  ValuePtr index_value = index_abs->BuildValue();
  if (index_value == kAnyValue) {
    abs.pop_back();
    return CheckAndConvertUtils::BroadenAllSequenceElements(std::make_shared<abstract::AbstractList>(abs));
  }

  int64_t index = 0;
  if (index_value->isa<Int64Imm>()) {
    index = GetValue<int64_t>(index_value);
  } else {
    index = IntToLong(GetValue<int32_t>(index_value));
  }
  if (index < -SizeToLong(elements.size()) || index >= SizeToLong(elements.size())) {
    MS_EXCEPTION(ValueError) << "The prim '" << prim_name << "', pop index[" << index << "] out of range.";
  }
  index = index < 0 ? index + SizeToLong(elements.size()) : index;
  abs.erase(abs.begin() + index);
  return std::make_shared<abstract::AbstractList>(abs);
}

class ListAppendAndInsertGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ListAppendAndInsertGradInnerInfer(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return ListAppendAndInsertGradInnerInfer(primitive, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ListAppendAndInsertGradInnerInfer(primitive, input_args);
  }
};
MIND_API_OPERATOR_IMPL(ListAppendAndInsertGrad, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ListAppendAndInsertGrad, prim::kPrimListAppendAndInsertGrad,
                                 ListAppendAndInsertGradInfer, false);
}  // namespace ops
}  // namespace mindspore
