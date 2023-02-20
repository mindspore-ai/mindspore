/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/xdivy.h"

#include <memory>
#include <string>
#include <vector>
#include <set>

#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "abstract/ops/op_infer.h"
#include "abstract/param_validator.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
const int64_t kXdivyInputNum = 2;
MIND_API_OPERATOR_IMPL(Xdivy, BaseOperator);
class XdivyInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kXdivyInputNum,
                                             prim_name);
    for (const auto &item : input_args) {
      MS_EXCEPTION_IF_NULL(item);
    }
    return BroadCastInferShape(prim_name, input_args);
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, kXdivyInputNum,
                                             prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    MS_EXCEPTION_IF_NULL(input_args[1]);
    auto x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
    auto y = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 1);
    (void)abstract::CheckDtypeSame(prim_name, x, y);
    auto input_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    if (!input_type->isa<TensorType>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name
                              << "', input must be a tensor, but got: " << input_type->ToString() << ".";
    }
    const std::set<TypePtr> valid_types = {kFloat32, kFloat16, kFloat64, kComplex64, kComplex128};
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x", input_type, valid_types, prim_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("y", input_type, valid_types, prim_name);
    return input_type;
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Xdivy, prim::kPrimXdivy, XdivyInfer, false);
}  // namespace ops
}  // namespace mindspore
