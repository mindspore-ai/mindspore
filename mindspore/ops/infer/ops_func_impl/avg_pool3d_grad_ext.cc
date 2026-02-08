/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/avg_pool3d_grad_ext.h"
#include <memory>
#include "primitive/op_name.h"

namespace mindspore {
namespace ops {
ShapeArray AvgPool3DGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex1]->GetShape()};
}

std::vector<TypeId> AvgPool3DGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
