/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/stack_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateStackParameter(const void *prim) {
  StackParameter *stack_param = reinterpret_cast<StackParameter *>(malloc(sizeof(StackParameter)));
  if (stack_param == nullptr) {
    MS_LOG(ERROR) << "malloc StackParameter failed.";
    return nullptr;
  }
  memset(stack_param, 0, sizeof(StackParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Stack();
  stack_param->op_parameter_.type_ = primitive->value_type();
  stack_param->axis_ = static_cast<int>(value->axis());
  return reinterpret_cast<OpParameter *>(stack_param);
}
}  // namespace
Registry g_stackParameterRegistry(schema::PrimitiveType_Stack, PopulateStackParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
