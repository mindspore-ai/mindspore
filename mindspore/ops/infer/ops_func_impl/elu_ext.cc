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

#include "infer/ops_func_impl/elu_ext.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"

namespace mindspore {
namespace ops {
ShapeArray EluExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(input_infos[0]);
  return {input_infos[0]->GetShape()};
}

std::vector<TypeId> EluExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(input_infos[0]);
  return {input_infos[0]->GetType()};
}

REGISTER_SIMPLE_INFER(kNameEluExt, EluExtFuncImpl)

}  // namespace ops
}  // namespace mindspore
