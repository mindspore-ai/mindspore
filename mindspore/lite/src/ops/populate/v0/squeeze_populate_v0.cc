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
#include "nnacl/squeeze_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSqueezeParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto squeeze_prim = primitive->value_as_Squeeze();
  if (squeeze_prim == nullptr) {
    MS_LOG(ERROR) << "squeeze_prim is nullptr";
    return nullptr;
  }
  auto *squeeze_param = reinterpret_cast<SqueezeParameter *>(malloc(sizeof(SqueezeParameter)));
  if (squeeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc SqueezeParameter failed.";
    return nullptr;
  }
  memset(squeeze_param, 0, sizeof(SqueezeParameter));
  squeeze_param->op_parameter_.type_ = schema::PrimitiveType_Squeeze;
  auto axis = squeeze_prim->axis();
  if (axis != nullptr) {
    squeeze_param->axis_size_ = axis->size();
    for (size_t i = 0; i < squeeze_param->axis_size_; i++) {
      squeeze_param->axis_[i] = *(axis->begin() + i);
    }
  } else {
    squeeze_param->axis_size_ = 0;
  }

  return reinterpret_cast<OpParameter *>(squeeze_param);
}
}  // namespace

Registry g_squeezeV0ParameterRegistry(schema::v0::PrimitiveType_Squeeze, PopulateSqueezeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
