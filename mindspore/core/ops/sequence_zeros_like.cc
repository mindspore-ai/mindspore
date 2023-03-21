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

#include "ops/sequence_zeros_like.h"

#include <vector>
#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
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
AbstractBasePtr MakeSequenceZeros(const abstract::AbstractSequencePtr &seq_abs) {
  if (seq_abs->dynamic_len()) {
    return seq_abs;
  }
  abstract::AbstractBasePtrList abs;
  const auto &seq_elements = seq_abs->elements();
  for (const auto &seq_element : seq_elements) {
    if (seq_element->isa<abstract::AbstractTensor>()) {
      (void)abs.emplace_back(TensorConstructUtils::CreateOnesTensor(
                               seq_element->BuildType(), seq_element->BuildShape()->cast<abstract::ShapePtr>()->shape())
                               ->ToAbstract());
    } else if (seq_element->isa<abstract::AbstractScalar>()) {
      (void)abs.emplace_back(std::make_shared<abstract::AbstractScalar>(MakeValue(0), seq_element->BuildType()));
    } else if (seq_element->isa<abstract::AbstractTuple>() || seq_element->isa<abstract::AbstractList>()) {
      (void)abs.emplace_back(MakeSequenceZeros(seq_element->cast<abstract::AbstractSequencePtr>()));
    } else {
      MS_EXCEPTION(TypeError) << "For 'SequenceZerosLike' is not supported " << seq_abs->BuildType()->ToString() << '.';
    }
  }
  if (seq_abs->isa<abstract::AbstractTuple>()) {
    return std::make_shared<abstract::AbstractTuple>(abs);
  }
  return std::make_shared<abstract::AbstractList>(abs);
}

AbstractBasePtr SequenceZerosLikeInferInner(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 1;
  constexpr size_t seq_index = 0;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto first_abs = input_args[seq_index];
  if (!first_abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the first input should be tuple or list but got: " << first_abs->ToString();
  }
  auto seq_abs = first_abs->cast<abstract::AbstractSequencePtr>();
  return MakeSequenceZeros(seq_abs);
}

MIND_API_OPERATOR_IMPL(SequenceZerosLike, BaseOperator);
class SequenceZerosLikeInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceZerosLikeInferInner(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceZerosLikeInferInner(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return SequenceZerosLikeInferInner(primitive, input_args);
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceZerosLike, prim::kPrimSequenceZerosLike, SequenceZerosLikeInfer, false);
}  // namespace ops
}  // namespace mindspore
