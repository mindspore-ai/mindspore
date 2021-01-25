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
#include "nnacl/fp32/one_hot_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateOneHotParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto one_hot_prim = primitive->value_as_OneHot();
  OneHotParameter *one_hot_param = reinterpret_cast<OneHotParameter *>(malloc(sizeof(OneHotParameter)));
  if (one_hot_param == nullptr) {
    MS_LOG(ERROR) << "malloc OneHotParameter failed.";
    return nullptr;
  }
  memset(one_hot_param, 0, sizeof(OneHotParameter));
  one_hot_param->op_parameter_.type_ = schema::PrimitiveType_OneHot;

  if (one_hot_prim == nullptr) {
    free(one_hot_param);
    MS_LOG(ERROR) << "get OneHot param nullptr.";
    return nullptr;
  }
  one_hot_param->axis_ = one_hot_prim->axis();
  return reinterpret_cast<OpParameter *>(one_hot_param);
}
}  // namespace

Registry g_oneHotV0ParameterRegistry(schema::v0::PrimitiveType_OneHot, PopulateOneHotParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
