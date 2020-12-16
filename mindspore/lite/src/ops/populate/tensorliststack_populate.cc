/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/tensorlist_stack.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/tensorlist_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateTensorListStackParameter(const mindspore::lite::PrimitiveC *primitive) {
  TensorListParameter *stack_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (stack_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(stack_param, 0, sizeof(TensorListParameter));
  stack_param->op_parameter_.type_ = primitive->Type();
  auto stack =
    reinterpret_cast<mindspore::lite::TensorListStack *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  stack_param->element_dtype_ = stack->GetElementDType();
  stack_param->num_element_ = stack->GetNumElements();
  return reinterpret_cast<OpParameter *>(stack_param);
}
Registry TensorListStackParameterRegistry(schema::PrimitiveType_TensorListStack, PopulateTensorListStackParameter);

}  // namespace lite
}  // namespace mindspore
