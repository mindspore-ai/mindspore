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
#include "ops/grad/prelu_grad.h"

#include <string>
#include <vector>
#include <memory>

#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/container.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(PReLUGrad, BaseOperator);
class PReLUGradInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    const int64_t kPReLUGradInputsNum = 3;
    const int64_t input_num = kPReLUGradInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    auto dx_shape = input_args[kInputIndex0]->BuildShape();
    MS_EXCEPTION_IF_NULL(dx_shape);
    auto dw_shape = input_args[kInputIndex2]->BuildShape();
    MS_EXCEPTION_IF_NULL(dw_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{dx_shape, dw_shape});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    const int64_t kPReLUGradInputsNum = 3;
    const int64_t input_num = kPReLUGradInputsNum;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
    auto dout = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
    auto out = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex1);
    (void)abstract::CheckDtypeSame(prim_name, out, dout);
    auto dx_type = input_args[kInputIndex0]->BuildType();
    MS_EXCEPTION_IF_NULL(dx_type);
    if (!dx_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << dx_type->ToString()
                              << ".";
    }
    auto dw_type = input_args[kInputIndex2]->BuildType();
    MS_EXCEPTION_IF_NULL(dw_type);
    if (!dw_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', input must be a Tensor, but got: " << dw_type->ToString()
                              << ".";
    }
    return std::make_shared<Tuple>(std::vector<TypePtr>{dx_type, dw_type});
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(PReLUGrad, prim::kPrimPReLUGrad, PReLUGradInfer, false);
}  // namespace ops
}  // namespace mindspore
