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

#include "infer/ops_func_impl/real_view.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace ops {
ShapeArray RealViewFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION(RuntimeError)
    << "RealViewFuncImpl::InferShape is not implemented,because it is not supported in Graph mode";
}

std::vector<TypeId> RealViewFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION(RuntimeError)
    << "RealViewFuncImpl::InferType is not implemented,because it is not supported in Graph mode";
}

REGISTER_SIMPLE_INFER(kNameRealView, RealViewFuncImpl)

}  // namespace ops
}  // namespace mindspore
