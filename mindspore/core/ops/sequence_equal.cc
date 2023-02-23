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

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#include "ops/tuple_equal.h"
#include "ops/list_equal.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SequenceEqualInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
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
  if (!x_abs->isa<abstract::AbstractSequence>() && !y_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For primitive '" << prim_name << "', the input must be a list or tuple, "
                            << "but got: " << x_abs->ToString() << " and " << y_abs->ToString();
  }
  auto seqx_abs = x_abs->cast<abstract::AbstractSequencePtr>();
  auto seqy_abs = y_abs->cast<abstract::AbstractSequencePtr>();
  if (seqx_abs->dynamic_len() || seqy_abs->dynamic_len() || seqx_abs->BuildValue() == kAnyValue ||
      seqy_abs->BuildValue() == kAnyValue) {
    return std::make_shared<abstract::AbstractScalar>(kAnyValue, kBool);
  }
  return std::make_shared<abstract::AbstractScalar>(*seqx_abs->BuildValue() == *seqy_abs->BuildValue());
}

class SequenceEqualInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceEqualInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceEqualInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceEqualInferInner(primitive, input_args);
  }
};
MIND_API_OPERATOR_IMPL(tuple_equal, BaseOperator);
MIND_API_OPERATOR_IMPL(list_equal, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(tuple_equal, prim::kPrimTupleEqual, SequenceEqualInfer, false);
REGISTER_PRIMITIVE_OP_INFER_IMPL(list_equal, prim::kPrimListEqual, SequenceEqualInfer, false);
}  // namespace ops
}  // namespace mindspore
