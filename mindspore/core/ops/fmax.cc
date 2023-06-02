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

#include "ops/fmax.h"
#include <memory>
#include <set>
#include <string>
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr FmaxInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  return BroadCastInferShape(prim_name, input_args);
}

TypePtr FmaxInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t kInputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, op_name);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt32, kInt64};
  auto x_type = input_args[0]->BuildType();
  auto y_type = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x1", x_type, valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x2", y_type, valid_types, op_name);
  return x_type;
}
}  // namespace
MIND_API_OPERATOR_IMPL(Fmax, BaseOperator);

class MIND_API FmaxInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return FmaxInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return FmaxInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Fmax, prim::kPrimFmax, FmaxInfer, false);
}  // namespace ops
}  // namespace mindspore
