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
#include <memory>
#include "src/ops/populate/populate_register.h"
#include "nnacl/reduce_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateReduceParameter(const void *prim) {
  ReduceParameter *reduce_param = reinterpret_cast<ReduceParameter *>(malloc(sizeof(ReduceParameter)));
  if (reduce_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReduceParameter failed.";
    return nullptr;
  }
  memset(reduce_param, 0, sizeof(ReduceParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_ReduceFusion();
  reduce_param->op_parameter_.type_ = primitive->value_type();
  reduce_param->keep_dims_ = value->keep_dims();
  reduce_param->reduce_to_end_ = value->reduce_to_end();
  reduce_param->coeff = value->coeff();
  reduce_param->mode_ = static_cast<int>(value->mode());
  return reinterpret_cast<OpParameter *>(reduce_param);
}

Registry ReduceParameterRegistry(schema::PrimitiveType_ReduceFusion, PopulateReduceParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
