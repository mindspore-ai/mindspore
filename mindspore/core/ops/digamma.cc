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

#include <memory>
#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/utils.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/digamma.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr DigammaInferShape(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape_ptr = CheckAndConvertUtils::GetTensorInputShape("Digamma", input_args, 0);
  auto input_shape = input_shape_ptr->shape();
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(input_shape);
  }
  if (input_shape.size() != 0 && input_shape[0] == 0) {
    MS_EXCEPTION(ValueError) << "For Digamma, the input must have value.";
  }
  return input_shape_ptr;
}
TypePtr DigammaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  auto input = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input, valid_types, prim_name);
  return input;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Digamma, BaseOperator);
// AG means auto generated
class MIND_API AGDigammaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return DigammaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return DigammaInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Digamma, prim::kPrimDigamma, AGDigammaInfer, false);
}  // namespace ops
}  // namespace mindspore
