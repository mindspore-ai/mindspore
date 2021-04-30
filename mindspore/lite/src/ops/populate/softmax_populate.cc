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
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore {
namespace lite {
OpParameter *PopulateSoftmaxParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Softmax();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SoftmaxParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto axis = value->axis();
  if (axis == nullptr) {
    MS_LOG(ERROR) << "axis is nullptr";
    free(param);
    return nullptr;
  }
  if (axis->size() != 1) {
    MS_LOG(ERROR) << "axis number invalid!number: " << axis->size();
    free(param);
    return nullptr;
  }
  param->axis_ = axis->data()[0];
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Softmax, PopulateSoftmaxParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
