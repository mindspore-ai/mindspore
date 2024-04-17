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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr SequenceMulInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto queue_shape = input_args[kIndex0]->GetShape()->cast<abstract::SequenceShapePtr>();
  MS_EXCEPTION_IF_NULL(queue_shape);
  abstract::BaseShapePtrList shape_elements = queue_shape->shape();
  abstract::BaseShapePtrList output_shape_list;
  auto scalar_value = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue()).value();
  for (int i = 0; i < scalar_value; i++) {
    output_shape_list.insert(output_shape_list.end(), shape_elements.begin(), shape_elements.end());
  }
  if (CheckAndConvertUtils::IsTuple(input_args[kIndex0])) {
    return std::make_shared<abstract::TupleShape>(output_shape_list);
  } else {
    return std::make_shared<abstract::ListShape>(output_shape_list);
  }
}

TypePtr SequenceMulInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  TypePtrList type_elements;
  bool isTuple = CheckAndConvertUtils::IsTuple(input_args[kIndex0]);

  if (isTuple) {
    auto queue_type = input_args[kIndex0]->GetType()->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(queue_type);
    type_elements = queue_type->elements();
  } else {
    auto queue_type = input_args[kIndex0]->GetType()->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(queue_type);
    type_elements = queue_type->elements();
  }
  TypePtrList output_type_list;
  auto scalar_value = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue()).value();
  for (int i = 0; i < scalar_value; i++) {
    output_type_list.insert(output_type_list.end(), type_elements.begin(), type_elements.end());
  }
  if (isTuple) {
    return std::make_shared<Tuple>(output_type_list);
  } else {
    return std::make_shared<List>(output_type_list);
  }
}

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
  (void)CheckAndConvertUtils::CheckTypeValid("scalar", scalar_abs->GetType(), scalar_valid_types, prim_name);
  if (seq_abs->dynamic_len()) {
    return seq_abs;
  }

  if (scalar_abs->GetValue()->ContainsValueAny()) {
    if (CheckAndConvertUtils::CheckContainNestedOrIrregularSequence(input_args)) {
      // Sequence ops with nested or irregular sequence input should be convert to PyExecute node.
      return std::make_shared<abstract::AbstractAny>();
    }
    auto ret = seq_abs->Clone()->cast<abstract::AbstractSequencePtr>();
    ret->CheckAndConvertToDynamicLenSequence();
    return ret;
  }

  abstract::AbstractBasePtrList mul;
  int scalar_value = GetValue<int64_t>(scalar_abs->GetValue());
  for (int i = 0; i < scalar_value; ++i) {
    for (size_t j = 0; j < seq_abs->size(); ++j) {
      mul.push_back(seq_abs->elements()[j]);
    }
  }
  if (first_abs->isa<abstract::AbstractList>()) {
    return std::make_shared<abstract::AbstractList>(mul);
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
    return SequenceMulInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceMulInferType(prim, input_args);
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
