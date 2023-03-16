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

#include "ops/sequence_index.h"

#include <vector>
#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(SequenceIndex, BaseOperator);
class SequenceIndexInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return kInt64;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    const size_t input_num = 2;
    auto prim_name = primitive->name();
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
    if (seq_abs->BuildValue() == kValueAny || target_value == kValueAny) {
      return nullptr;
    }
    const auto &seq_elements = seq_abs->elements();
    for (size_t i = 0; i < seq_elements.size(); ++i) {
      auto element = seq_elements[i];
      if (CheckAndConvertUtils::CheckValueSame(target_value, element->BuildValue())) {
        return MakeValue(static_cast<int64_t>(i));
      }
    }
    MS_EXCEPTION(ValueError) << target_value->ToString() << " is not in " << seq_abs->ToString();
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(SequenceIndex, prim::kPrimSequenceIndex, SequenceIndexInfer, true);
}  // namespace ops
}  // namespace mindspore
