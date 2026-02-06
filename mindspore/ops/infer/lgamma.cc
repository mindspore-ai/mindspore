/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "infer/lgamma.h"

#include <set>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/tensor_type.h"
#include "ir/dtype/type.h"
#include "mindapi/base/type_id.h"
#include "mindapi/helper.h"
#include "primitive/math_ops.h"
#include "primitive/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr LgammaInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgsType(prim_name, input_args, 0, kObjectTypeTensorType);
  auto x = input_args[0]->GetShape();
  return x->Clone();
}

TypePtr LgammaInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt32};
  auto x_type = input_args[kInputIndex0]->GetType();
  TypeId tensor_type_id = CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim_name)->type_id();
  if (tensor_type_id == kNumberTypeInt32) {
    return std::make_shared<TensorType>(kFloat32);
  } else {
    return input_args[kInputIndex0]->GetType();
  }
}
}  // namespace

MIND_API_OPERATOR_IMPL(Lgamma, BaseOperator);

// AG means auto generated
class OPS_API AGLgammaInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return LgammaInferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return LgammaInferType(primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(Lgamma, prim::kPrimLgamma, AGLgammaInfer, false);
}  // namespace ops
}  // namespace mindspore
