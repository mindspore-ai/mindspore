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

#include "infer/ops_func_impl/addbmm.h"
#include <vector>
#include <memory>
#include "primitive/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

ShapeArray AddbmmFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &batch1 = input_infos[kInputIndex1];
  const auto &batch2 = input_infos[kInputIndex2];
  const auto &batch1_shape = input_infos[kInputIndex1]->GetShape();
  const auto &batch2_shape = input_infos[kInputIndex2]->GetShape();
  if (batch1->IsDynamicRank() || batch2->IsDynamicRank()) {
    ShapeVector ret_shape = {abstract::Shape::kShapeDimAny};
    return {ret_shape};
  }

  bool dynamic_shape = batch1->IsDynamic() || batch2->IsDynamic();
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
  ShapeVector ret_shape{batch1_shape[kDim1], batch2_shape[kDim2]};
  return {ret_shape};
}

std::vector<TypeId> AddbmmFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  const auto &input_type = input_infos[kInputIndex0]->GetType();
  return {input_type};
}

}  // namespace ops
}  // namespace mindspore
