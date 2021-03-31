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
#include "nnacl/softmax_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSoftmaxParameter(const void *prim) {
  SoftmaxParameter *softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  if (softmax_param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxParameter failed.";
    return nullptr;
  }
  memset(softmax_param, 0, sizeof(SoftmaxParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  softmax_param->op_parameter_.type_ = primitive->value_type();
  auto prim_softmax = primitive->value_as_Softmax();
  if (prim_softmax->axis()->size() != 1) {
    MS_LOG(ERROR) << "axis number invalid!number: " << prim_softmax->axis()->size();
    free(softmax_param);
    return nullptr;
  }
  softmax_param->axis_ = prim_softmax->axis()->data()[0];
  return reinterpret_cast<OpParameter *>(softmax_param);
}
}  // namespace

Registry g_softmaxParameterRegistry(schema::PrimitiveType_Softmax, PopulateSoftmaxParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
