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
#include "infer/ops_func_impl/swiglu_grad.h"
#include "abstract/dshape.h"
#include "primitive/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {
ShapeArray SwigluGradFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  ShapeVector shape = input_infos[kInputIndex1]->GetShape();
  return {shape};
}
std::vector<TypeId> SwigluGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  TypeId input_grad_type_id = input_infos[kIndex0]->GetType();
  TypeId input_x_type_id = input_infos[kIndex1]->GetType();
  if (input_grad_type_id != input_x_type_id) {
    MS_LOG_EXCEPTION << "For " << primitive->name()
                     << ", the grad type must be same as input type, but got input_grad_type: "
                     << TypeIdToString(input_grad_type_id) << " and input_x_type: " << TypeIdToString(input_x_type_id);
  }
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  CheckAndConvertUtils::CheckTypeIdValid("input_x", input_x_type_id, valid_types, primitive->name());
  return {input_x_type_id};
}
REGISTER_SIMPLE_INFER(kNameSwigluGrad, SwigluGradFuncImpl)

}  // namespace ops
}  // namespace mindspore
