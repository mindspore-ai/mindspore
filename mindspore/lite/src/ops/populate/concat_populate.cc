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
#include "nnacl/concat_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateConcatParameter(const void *prim) {
  ConcatParameter *concat_param = reinterpret_cast<ConcatParameter *>(malloc(sizeof(ConcatParameter)));
  if (concat_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConcatParameter failed.";
    return nullptr;
  }
  memset(concat_param, 0, sizeof(ConcatParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  concat_param->op_parameter_.type_ = primitive->value_type();
  concat_param->axis_ = static_cast<int>(primitive->value_as_Concat()->axis());
  return reinterpret_cast<OpParameter *>(concat_param);
}
}  // namespace

Registry g_concatParameterRegistry(schema::PrimitiveType_Concat, PopulateConcatParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
