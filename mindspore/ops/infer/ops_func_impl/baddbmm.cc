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

#include "infer/ops_func_impl/baddbmm.h"
#include <vector>
#include <memory>
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {
BaseShapePtr BaddbmmFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto batch1_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto batch1_shape = batch1_shape_ptr->GetShapeVector();
  auto batch2_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto batch2_shape = batch2_shape_ptr->GetShapeVector();
  if (IsDynamicRank(batch1_shape) || IsDynamicRank(batch2_shape)) {
    ShapeVector ret_shape = {abstract::Shape::kShapeDimAny};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  bool dynamic_shape = IsDynamic(batch1_shape) || IsDynamic(batch2_shape);
  if (!dynamic_shape) {
    if (batch1_shape.size() != kShape3dDims) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', input 'batch1' must be a 3D Tensor, but got:" << batch1_shape.size();
    }

    if (batch2_shape.size() != kShape3dDims) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name()
                        << "', input 'batch2' must be a 3D Tensor, but got:" << batch2_shape.size();
    }

    if (batch1_shape[kDim2] != batch2_shape[kDim1]) {
      MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', first dimension of 'batch2' must be equal to 'batch1' "
                        << batch1_shape[kDim2] << " , but got:" << batch2_shape[kDim1];
    }
  }
  ShapeVector ret_shape{batch1_shape[kDim0], batch1_shape[kDim1], batch2_shape[kDim2]};
  return std::make_shared<abstract::TensorShape>(ret_shape);
}

TypePtr BaddbmmFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}

TypePtrList BaddbmmFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}

ShapeArray BaddbmmFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &batch1_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(batch1_tensor);
  const auto &batch2_tensor = input_values[kIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(batch2_tensor);
  const auto &batch1_shape = batch1_tensor->shape();
  const auto &batch2_shape = batch2_tensor->shape();
  if (batch1_shape.size() != kShape3dDims) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', input 'batch1' must be a 3D Tensor, but got:" << batch1_shape.size();
  }

  if (batch2_shape.size() != kShape3dDims) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', input 'batch2' must be a 3D Tensor, but got:" << batch2_shape.size();
  }

  if (batch1_shape[kDim2] != batch2_shape[kDim1]) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', first dimension of 'batch2' must be equal to 'batch1' "
                      << batch1_shape[kDim2] << " , but got:" << batch2_shape[kDim1];
  }
  ShapeVector ret_shape{batch1_shape[kDim0], batch1_shape[kDim1], batch2_shape[kDim2]};
  return {ret_shape};
}

REGISTER_SIMPLE_INFER(kNameBaddbmm, BaddbmmFuncImpl)
}  // namespace ops
}  // namespace mindspore
