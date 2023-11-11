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

#include "ops/ops_func_impl/baddbmm.h"
#include <vector>
#include <memory>
#include "ops/op_name.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kMatSize = 3;
}
BaseShapePtr BaddbmmFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_args.size() != 5) {
    MS_LOG(EXCEPTION) << "input args size should be 5, but got " << input_args.size();
  }
  auto input_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto input_shape = input_shape_ptr->GetShapeVector();
  auto batch1_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto batch1_shape = batch1_shape_ptr->GetShapeVector();
  auto batch2_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto batch2_shape = batch2_shape_ptr->GetShapeVector();
  if (batch1_shape.size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', input 'batch1' must be a 3D Tensor, but got:" << batch1_shape.size();
  }

  if (batch2_shape.size() != kMatSize) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', input 'batch2' must be a 3D Tensor, but got:" << batch2_shape.size();
  }

  if (batch1_shape[2] != batch2_shape[1]) {
    MS_LOG(EXCEPTION) << "For '" << primitive->name() << "', first dimension of 'batch2' must be equal to 'batch1' "
                      << batch1_shape[2] << " , but got:" << batch2_shape[1];
  }
  ShapeVector ret_shape{batch1_shape[0], batch1_shape[1], batch2_shape[2]};
  return std::make_shared<abstract::TensorShape>(ret_shape);
}

TypePtr BaddbmmFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  return input_type->Clone();
}
}  // namespace ops
}  // namespace mindspore
