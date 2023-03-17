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

#include "ops/grad/shape_mul_grad.h"

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
AbstractBasePtr ShapeMulGradInnerInfer(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  constexpr size_t input_len = 2;
  constexpr size_t data_index = 0;
  constexpr size_t dout_index = 1;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, prim_name);
  auto data_abs = dyn_cast<abstract::AbstractTuple>(input_args[data_index]);
  MS_EXCEPTION_IF_NULL(data_abs);
  auto dout_abs = abstract::CheckArg<abstract::AbstractScalar>(prim_name, input_args, dout_index);
  if (!data_abs->isa<abstract::AbstractTuple>()) {
    MS_EXCEPTION(TypeError) << "The prim '" << prim_name
                            << "', the input_data must be list, dout must be scalar, but got " << data_abs->ToString()
                            << " dout is " << dout_abs->ToString();
  }

  if (data_abs->dynamic_len()) {
    return data_abs->Clone();
  }
  return CheckAndConvertUtils::BroadenAllSequenceElements(data_abs);
}

class ShapeMulGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return ShapeMulGradInnerInfer(primitive, input_args)->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    return ShapeMulGradInnerInfer(prim, input_args)->BuildType();
  }

  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return ShapeMulGradInnerInfer(primitive, input_args);
  }
};
MIND_API_OPERATOR_IMPL(ShapeMulGrad, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ShapeMulGrad, prim::kPrimShapeMulGrad, ShapeMulGradInfer, false);
}  // namespace ops
}  // namespace mindspore
