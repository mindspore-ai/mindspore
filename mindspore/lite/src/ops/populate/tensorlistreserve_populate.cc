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
OpParameter *PopulateTensorListReserveParameter(const void *prim) {
  TensorListParameter *reserve_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (reserve_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(reserve_param, 0, sizeof(TensorListParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_TensorListReserve();
  reserve_param->op_parameter_.type_ = primitive->value_type();
  reserve_param->element_dtype_ = value->element_dtype();
  return reinterpret_cast<OpParameter *>(reserve_param);
}
Registry TensorListReserveParameterRegistry(schema::PrimitiveType_TensorListReserve, PopulateTensorListReserveParameter,
                                            SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
