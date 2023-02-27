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
template <typename T>
AbstractBasePtr MakeZeros(const size_t &len) {
  abstract::AbstractBasePtrList abs;
  for (size_t i = 0; i < len; i++) {
    abs.push_back(std::make_shared<abstract::AbstractScalar>(T(0)));
  }
  return std::make_shared<abstract::AbstractTuple>(abs);
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
  if (seq_abs->dynamic_len()) {
    return seq_abs;
  }
  const auto &seq_elements = seq_abs->elements();
  const auto &len = seq_elements.size();
  auto type = seq_elements[0]->BuildType();
  if (type->type_id() == kInt64->type_id()) {
    return MakeZeros<int64_t>(len);
  } else if (type->type_id() == kInt32->type_id()) {
    return MakeZeros<int>(len);
  } else if (type->type_id() == kFloat32->type_id()) {
    return MakeZeros<float>(len);
  } else if (type->type_id() == kFloat64->type_id()) {
    return MakeZeros<double>(len);
  } else {
    MS_EXCEPTION(TypeError) << "For '" << prim_name << "' is not supported" << type->ToString() << '.';
  }
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
