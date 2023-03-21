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

#include "ops/shape_mul.h"

#include <vector>
#include <memory>
#include <string>

#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
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
class ShapeMulInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    constexpr size_t input_num = 1;
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    auto shape_x = input_args[0];
    MS_EXCEPTION_IF_NULL(shape_x);
    if (!shape_x->isa<abstract::AbstractTuple>()) {
      MS_EXCEPTION(TypeError) << "For primitive '" << prim_name
                              << "', the first input must be a tuple but got: " << shape_x->ToString();
    }
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return kInt64;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    constexpr size_t input_num = 1;
    auto prim_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    abstract::AbstractTuplePtr shape_x = abstract::CheckArg<abstract::AbstractTuple>(prim_name, input_args, 0);
    auto shpx_value = shape_x->BuildValue();
    if (shape_x->dynamic_len() || shape_x->BuildValue() == kValueAny) {
      return nullptr;
    }
    auto shpx_data = shpx_value->cast<ValueTuplePtr>()->value();
    int64_t result = 1;
    for (size_t i = 0; i < shpx_data.size(); i++) {
      int64_t value = GetValue<int64_t>(shpx_data[i]);
      result = IntMulWithOverflowCheck(result, value);
    }

    return MakeValue(result);
  }
};
MIND_API_OPERATOR_IMPL(shape_mul, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(shape_mul, prim::kPrimShapeMul, ShapeMulInfer, true);
}  // namespace ops
}  // namespace mindspore
