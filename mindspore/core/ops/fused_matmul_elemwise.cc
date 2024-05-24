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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/math_ops.h"
#include "ops/base_operator.h"
#include "ops/op_name.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {

TypePtr FusedMatMulElemBinary::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  constexpr auto kMatMulInputNum = 2;
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                           kMatMulInputNum, op_name);
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);

  auto x_tensor_type = x->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  auto y_tensor_type = y->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(y_tensor_type);
  TypePtr x_type = x_tensor_type->element();
  TypePtr y_type = y_tensor_type->element();

  if (x_type->type_id() != y_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the type of 'x2' should be same as 'x1', but got 'x1' with type Tensor["
                            << x_type->ToString() << "] and 'x2' with type Tensor[" << y_type->ToString() << "].";
  }
  if (primitive->HasAttr("cast_type")) {
    auto out_type = primitive->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "MatMul cast_type must be a `Type`";
    }
    x_type = out_type->cast<TypePtr>();
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::set<TypePtr> valid_types;
  valid_types = {kInt8, kFloat16, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  if (x_type->type_id() == TypeId::kNumberTypeInt8 && device_target == kAscendDevice) {
    return kInt32;
  }
  return x_type;
}

TypePtr FusedMatMulElemUnary::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  constexpr auto kMatMulInputNum = 2;
  auto op_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual,
                                           kMatMulInputNum, op_name);
  auto x = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 0, kObjectTypeTensorType);
  auto y = CheckAndConvertUtils::CheckArgsType(op_name, input_args, 1, kObjectTypeTensorType);

  auto x_tensor_type = x->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(x_tensor_type);
  auto y_tensor_type = y->GetType()->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(y_tensor_type);
  TypePtr x_type = x_tensor_type->element();
  TypePtr y_type = y_tensor_type->element();

  if (x_type->type_id() != y_type->type_id()) {
    MS_EXCEPTION(TypeError) << "For '" << op_name
                            << "', the type of 'x2' should be same as 'x1', but got 'x1' with type Tensor["
                            << x_type->ToString() << "] and 'x2' with type Tensor[" << y_type->ToString() << "].";
  }
  if (primitive->HasAttr("cast_type")) {
    auto out_type = primitive->GetAttr("cast_type");
    MS_EXCEPTION_IF_NULL(out_type);
    if (!out_type->isa<Type>()) {
      MS_EXCEPTION(ValueError) << "MatMul cast_type must be a `Type`";
    }
    x_type = out_type->cast<TypePtr>();
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::set<TypePtr> valid_types;
  valid_types = {kInt8, kFloat16, kBFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  if (x_type->type_id() == TypeId::kNumberTypeInt8 && device_target == kAscendDevice) {
    return kInt32;
  }
  return x_type;
}

}  // namespace ops
}  // namespace mindspore
