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

#include "src/ops/p_relu.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/prelu_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulatePReLUParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto param = reinterpret_cast<mindspore::lite::PReLU *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  PReluParameter *prelu_param = reinterpret_cast<PReluParameter *>(malloc(sizeof(PReluParameter)));
  if (prelu_param == nullptr) {
    MS_LOG(ERROR) << "malloc PReluParameter failed.";
    return nullptr;
  }
  memset(prelu_param, 0, sizeof(PReluParameter));
  prelu_param->op_parameter_.type_ = primitive->Type();
  prelu_param->channelShared = param->GetChannelShared();
  return reinterpret_cast<OpParameter *>(prelu_param);
}
Registry PReLUParameterRegistry(schema::PrimitiveType_PReLU, PopulatePReLUParameter);

}  // namespace lite
}  // namespace mindspore
