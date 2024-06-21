/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ops/fused_matmul_elemwise.h"
#include <memory>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
TypePtr FusedMatMulElemInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                 const int input_num) {
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto x_tensor_type = x->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  TypePtr x_type = x_tensor_type->element();
  if (x_type->type_id() == TypeId::kNumberTypeInt8) {
    return kFloat16;
  }
  return x_type;
}
}  // namespace

TypePtr FusedMatMulElemBinary::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  const int binary_input_num = 3;
  return FusedMatMulElemInferType(primitive, input_args, binary_input_num);
}

TypePtr FusedMatMulElemUnary::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  const int unary_input_num = 2;
  return FusedMatMulElemInferType(primitive, input_args, unary_input_num);
}
}  // namespace ops
}  // namespace mindspore
