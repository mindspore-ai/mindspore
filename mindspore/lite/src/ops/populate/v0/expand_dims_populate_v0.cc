/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/expandDims_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateExpandDimsParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto expand_dims_prim = primitive->value_as_ExpandDims();

  ExpandDimsParameter *expand_dims_param = reinterpret_cast<ExpandDimsParameter *>(malloc(sizeof(ExpandDimsParameter)));
  if (expand_dims_param == nullptr) {
    MS_LOG(ERROR) << "malloc ExpandDimsParameter failed.";
    return nullptr;
  }
  memset(expand_dims_param, 0, sizeof(ExpandDimsParameter));
  expand_dims_param->op_parameter_.type_ = schema::PrimitiveType_ExpandDims;
  expand_dims_param->dim_ = expand_dims_prim->dim();
  return reinterpret_cast<OpParameter *>(expand_dims_param);
}
}  // namespace

Registry g_expandDimsV0ParameterRegistry(schema::v0::PrimitiveType_ExpandDims, PopulateExpandDimsParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
