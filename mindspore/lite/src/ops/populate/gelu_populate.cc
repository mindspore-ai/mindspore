/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "src/ops/gelu.h"
#include "nnacl/gelu_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateGeLUParameter(const mindspore::lite::PrimitiveC *primitive) {
  //  const auto param = reinterpret_cast<mindspore::lite::GeLU *>(const_cast<mindspore::lite::PrimitiveC
  //  *>(primitive));
  GeLUParameter *gelu_param = reinterpret_cast<GeLUParameter *>(malloc(sizeof(GeLUParameter)));
  if (gelu_param == nullptr) {
    MS_LOG(ERROR) << "malloc GeLUParameter failed.";
    return nullptr;
  }
  memset(gelu_param, 0, sizeof(GeLUParameter));
  gelu_param->op_parameter_.type_ = primitive->Type();
  //  gelu_param->approximate_ = param->GetApproximate();
  return reinterpret_cast<OpParameter *>(gelu_param);
}

// Registry GeLUParameterRegistry(schema::PrimitiveType_GeLU, PopulateGeLUParameter);
}  // namespace lite
}  // namespace mindspore
