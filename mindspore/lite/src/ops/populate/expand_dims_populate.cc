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
#include "nnacl/fp32/expandDims_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateExpandDimsParameter(const void *prim) {
  ExpandDimsParameter *expand_param = reinterpret_cast<ExpandDimsParameter *>(malloc(sizeof(ExpandDimsParameter)));
  if (expand_param == nullptr) {
    MS_LOG(ERROR) << "malloc ExpandDimsParameter failed.";
    return nullptr;
  }
  memset(expand_param, 0, sizeof(ExpandDimsParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  expand_param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(expand_param);
}
}  // namespace

Registry g_expandDimsParameterRegistry(schema::PrimitiveType_ExpandDims, PopulateExpandDimsParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
