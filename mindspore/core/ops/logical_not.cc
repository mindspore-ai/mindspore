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

#include <set>

#include "ops/logical_not.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr LogicalNotInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto LogicalNot_prim = primitive->cast<PrimLogicalNotPtr>();
  MS_EXCEPTION_IF_NULL(LogicalNot_prim);
  auto op_name = LogicalNot_prim->name();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), op_name);
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr LogicalNotInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto LogicalNot_prim = prim->cast<PrimLogicalNotPtr>();
  MS_EXCEPTION_IF_NULL(LogicalNot_prim);
  auto op_name = LogicalNot_prim->name();
  auto infer_dtype = input_args[0]->BuildType();
  std::set<TypeId> local_bool = {kNumberTypeBool};
  CheckAndConvertUtils::CheckTensorTypeValid("x", infer_dtype, local_bool, op_name);
  auto tensor_type = infer_dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto element = tensor_type->element();
  return element;
}
}  // namespace
AbstractBasePtr LogicalNotInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(LogicalNotInferType(primitive, input_args),
                                                    LogicalNotInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameLogicalNot, LogicalNot);
}  // namespace ops
}  // namespace mindspore
