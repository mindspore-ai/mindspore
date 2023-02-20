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

#include "ops/tensor_to_scalar.h"

#include <vector>
#include <string>
#include <memory>

#include "utils/check_convert_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
class TensorToScalarInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, op_name);

    auto shape_ptr = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 0);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto x_shape = shape_ptr->shape();
    if (!x_shape.empty() && !IsDynamic(x_shape)) {
      MS_EXCEPTION(ValueError) << "For Primitive[" << op_name << "], the input shape must be empty, but got " << x_shape
                               << ".";
    }
    return abstract::kNoShape;
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    constexpr size_t input_len = 1;
    constexpr size_t input_0_index = 0;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, op_name);
    auto x_type = input_args[input_0_index]->BuildType();
    MS_EXCEPTION_IF_NULL(x_type);
    if (!x_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For Primitive[" << op_name << "], the input must be a Tensor but got "
                              << x_type->ToString() << ".";
    }
    auto tensor_type = x_type->cast<TensorTypePtr>();
    auto element = tensor_type->element();
    MS_EXCEPTION_IF_NULL(element);
    return element;
  }
};
MIND_API_OPERATOR_IMPL(TensorToScalar, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(TensorToScalar, prim::kPrimTensorToScalar, TensorToScalarInfer, false);
}  // namespace ops
}  // namespace mindspore
