/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "ops/ops_func_impl/bmm.h"
#include <vector>
#include <memory>
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BmmFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != kSize2) {
    MS_LOG(EXCEPTION) << "input args size should be 2, but got " << input_args.size();
  }
  auto prim_name = primitive->name();
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto mat2_shape_ptr = input_args[kInputIndex1]->GetShape();

  MS_EXCEPTION_IF_NULL(input_shape_ptr);
  MS_EXCEPTION_IF_NULL(mat2_shape_ptr);

  auto input_shape = input_shape_ptr->GetShapeVector();
  auto mat2_shape = mat2_shape_ptr->GetShapeVector();
  if (input_shape.size() != kShape3dDims) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'input' must be a 3D Tensor, but got:" << input_shape.size();
  }

  if (mat2_shape.size() != kShape3dDims) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'mat2' must be a 3D Tensor, but got:" << mat2_shape.size();
  }

  if (input_shape[kDim0] != mat2_shape[kDim0]) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', first dimension of 'mat2' must be equal to 'input' "
                      << input_shape[kDim0] << " , but got:" << mat2_shape[kDim0];
  }

  if (input_shape[kDim2] != mat2_shape[kDim1]) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', first dimension of 'batch2' must be equal to 'batch1' "
                      << input_shape[kDim2] << " , but got:" << mat2_shape[kDim1];
  }

  ShapeVector ret_shape{input_shape[kDim0], input_shape[kDim1], mat2_shape[kDim2]};
  return std::make_shared<abstract::TensorShape>(ret_shape);
}

TypePtr BmmFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
