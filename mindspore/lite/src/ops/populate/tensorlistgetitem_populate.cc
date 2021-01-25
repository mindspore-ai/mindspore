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
#include "nnacl/tensorlist_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateTensorListGetItemParameter(const void *prim) {
  TensorListParameter *getItem_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (getItem_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(getItem_param, 0, sizeof(TensorListParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_TensorListGetItem();
  getItem_param->op_parameter_.type_ = primitive->value_type();
  getItem_param->element_dtype_ = value->element_dtype();
  return reinterpret_cast<OpParameter *>(getItem_param);
}
Registry TensorListGetItemParameterRegistry(schema::PrimitiveType_TensorListGetItem, PopulateTensorListGetItemParameter,
                                            SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
