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

#include "src/ops/softmax.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/softmax_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSoftmaxParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto softmax_primitive =
    reinterpret_cast<mindspore::lite::SoftMax *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  SoftmaxParameter *softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  if (softmax_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxParameter failed.";
    return nullptr;
  }
  memset(softmax_param, 0, sizeof(SoftmaxParameter));
  softmax_param->op_parameter_.type_ = primitive->Type();
  softmax_param->axis_ = softmax_primitive->GetAxis();
  return reinterpret_cast<OpParameter *>(softmax_param);
}

Registry SoftMaxParameterRegistry(schema::PrimitiveType_SoftMax, PopulateSoftmaxParameter);

}  // namespace lite
}  // namespace mindspore
