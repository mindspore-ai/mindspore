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

#include "ops/fmin.h"

#include <memory>
#include <set>
#include <vector>

#include "abstract/ops/op_infer.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(Fmin, BaseOperator);
class FminInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    return BroadCastInferShape(prim_name, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    auto op_name = primitive->name();
    const int64_t kInputNum = 2;
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, op_name);
    const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
    auto type_x = input_args[0]->BuildType();
    auto type_y = input_args[1]->BuildType();
    MS_EXCEPTION_IF_NULL(type_x);
    MS_EXCEPTION_IF_NULL(type_y);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", type_x, valid_types, op_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", type_y, valid_types, op_name);
    return type_x;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(Fmin, prim::kPrimFmin, FminInfer, false);
}  // namespace ops
}  // namespace mindspore
