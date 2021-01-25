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
#include "nnacl/squeeze_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSqueezeParameter(const void *prim) {
  SqueezeParameter *squeeze_param = reinterpret_cast<SqueezeParameter *>(malloc(sizeof(SqueezeParameter)));
  if (squeeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc SqueezeParameter failed.";
    return nullptr;
  }
  memset(squeeze_param, 0, sizeof(SqueezeParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  squeeze_param->op_parameter_.type_ = primitive->value_type();

  auto squeeze_prim = primitive->value_as_Squeeze();
  if (squeeze_prim->axis() != nullptr) {
    squeeze_param->axis_size_ = squeeze_prim->axis()->size();
    for (size_t i = 0; i < squeeze_param->axis_size_; i++) {
      squeeze_param->axis_[i] = *(squeeze_prim->axis()->begin() + i);
    }
  } else {
    squeeze_param->axis_size_ = 0;
  }

  return reinterpret_cast<OpParameter *>(squeeze_param);
}
}  // namespace
Registry g_squeezeParameterRegistry(schema::PrimitiveType_Squeeze, PopulateSqueezeParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
