/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/tril_ext.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ops {
BaseShapePtr TrilExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_vec = input_args[0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(input_shape_vec))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  const int64_t kMinShapeSize = 2;
  auto input_shape_rank = SizeToLong(input_shape_vec.size());
  (void)CheckAndConvertUtils::CheckInteger("input's rank", input_shape_rank, kGreaterEqual, kMinShapeSize,
                                           primitive->name());
  return std::make_shared<abstract::Shape>(input_shape_vec);
}

TypePtr TrilExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, common_valid_types_with_bool,
                                                   primitive->name());
  return input_type;
}

ShapeArray TrilExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto &input_shape_vec = x_tensor->shape();
  const int64_t kMinShapeSize = 2;
  auto input_shape_rank = SizeToLong(input_shape_vec.size());
  (void)CheckAndConvertUtils::CheckInteger("input's rank", input_shape_rank, kGreaterEqual, kMinShapeSize,
                                           primitive->name());
  return {x_tensor->shape()};
}

TypePtrList TrilExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto input_type = x_tensor->Dtype();
  return {input_type};
}

REGISTER_SIMPLE_INFER(kNameTrilExt, TrilExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
