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

#include "infer/ops_func_impl/mv.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include "primitive/op_name.h"
#include "abstract/dshape.h"
#include "ir/primitive.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray MvFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto vec_shape = input_infos[kIndex1]->GetShape();

  if (input_infos[kIndex0]->IsDynamicRank() || input_infos[kIndex1]->IsDynamicRank()) {
    // The returned rank must be 1D, so the infer shape is -1.
    return {{abstract::Shape::kShapeDimAny}};
  }
  if (!input_infos[kIndex0]->IsDynamic() && !input_infos[kIndex1]->IsDynamic()) {
    // The dimension of input tensor must be 2D, and vec tensor must be 1D.
    MS_ASSERT_TRUE(input_shape.size() == kDim2)
      << "For '" << primitive->name() << "', the input 'input' must be a 2D dimensional Tensor, but got "
      << input_shape.size() << "D shape " << input_shape;
    MS_ASSERT_TRUE(vec_shape.size() == kDim1)
      << "For '" << primitive->name() << "', the input 'vec' must be a 1D dimensional Tensor, but got "
      << vec_shape.size() << "D shape " << vec_shape;
    // input_shape and vec_shape must meet the vector multiplication rules.
    MS_ASSERT_TRUE(input_shape[1] == vec_shape[0])
      << "For " << primitive->name()
      << ", the row of the input 'input' should be same as the elements of the input 'vec', with input shape "
      << input_shape << ", vec shape " << vec_shape;
  }
  return {{input_shape[0]}};
}

std::vector<TypeId> MvFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
