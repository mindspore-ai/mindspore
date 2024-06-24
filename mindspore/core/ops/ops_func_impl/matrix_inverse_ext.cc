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
#include <memory>
#include <set>
#include <string>

#include "ops/ops_func_impl/matrix_inverse_ext.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops/op_utils.h"

namespace {
const constexpr int64_t kNumber1 = 1;
const constexpr int64_t kNumber2 = 2;
}  // namespace
namespace mindspore {
namespace ops {

void CheckMatrixInverseExtShape(const ShapeVector &input_shape, const std::string &prim_name) {
  auto input_rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(input_rank != 1, "For 'MatrixInverseExt', the input tensor must have at least 2 dimensions");
  if (input_rank >= kNumber2) {
    MS_CHECK_VALUE(input_shape[input_rank - kNumber1] == input_shape[input_rank - kNumber2],
                   "For 'MatrixInverseExt', the last two dimensions should be same");
  }

  MS_CHECK_VALUE(input_rank <= 6, "For 'MatrixInverseExt', the input tensor cannot be larger than 6 dimensions.");
}

BaseShapePtr MatrixInverseExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape()->cast<abstract::ShapePtr>();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  if (!x_shape_ptr->IsDynamic()) {
    CheckMatrixInverseExtShape(x_shape, primitive->name());
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MatrixInverseExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  const std::set<TypePtr> valid_types = {kFloat32};
  auto infer_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", infer_type, valid_types, primitive->name());
  return infer_type;
}

TypePtrList MatrixInverseExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_CHECK_VALUE(x_tensor->data_type() == kNumberTypeFloat32,
                 "For Primitive [MatrixInverseExt], type should be float32");
  const auto &input_type = x_tensor->Dtype();
  return {input_type};
}
ShapeArray MatrixInverseExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  CheckMatrixInverseExtShape(x_tensor->shape(), primitive->name());
  return {x_tensor->shape()};
}

REGISTER_SIMPLE_INFER(kNameMatrixInverseExt, MatrixInverseExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
