/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "ops/sequence_mul.h"

#include <vector>
#include <memory>
#include <set>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
AbstractBasePtr SequenceMulInferInner(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t seq_index = 0;
  constexpr size_t scalar_index = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto first_abs = input_args[seq_index];
  if (!first_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the first input should be tuple or list but got: " << first_abs->ToString();
  }
  auto seq_abs = first_abs->cast<abstract::AbstractSequencePtr>();
  auto scalar_abs = input_args[scalar_index];
  const std::set<TypePtr> scalar_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("scalar", scalar_abs->BuildType(), scalar_valid_types, prim_name);
  if (seq_abs->dynamic_len()) {
    return seq_abs;
  }

  if (scalar_abs->BuildValue() == kAnyValue) {
    auto ret = seq_abs->Clone()->cast<abstract::AbstractSequencePtr>();
    ret->CheckAndConvertToDynamicLenSequence();
    return ret;
  }

  abstract::AbstractBasePtrList mul;
  int scalar_value = GetValue<int64_t>(scalar_abs->BuildValue());
  for (int i = 0; i < scalar_value; ++i) {
    for (size_t j = 0; j < seq_abs->size(); ++j) {
      mul.push_back(seq_abs->elements()[j]);
    }
  }
  auto ret = std::make_shared<abstract::AbstractTuple>(mul);
  return ret;
}
}  // namespace

MIND_API_OPERATOR_IMPL(SequenceMul, BaseOperator);
class SequenceMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMulInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMulInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMulInferInner(primitive, input_args);
  }

  std::set<int64_t> GetValueDependArgIndices() const override { return {1}; }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceMul, prim::kPrimSequenceMul, SequenceMulInfer, false);
}  // namespace ops
}  // namespace mindspore
