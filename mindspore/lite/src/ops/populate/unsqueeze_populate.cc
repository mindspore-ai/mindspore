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
#include "nnacl/unsqueeze_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateUnsqueezeParameter(const void *prim) {
  UnSqueezeParameter *unsqueeze_param = reinterpret_cast<UnSqueezeParameter *>(malloc(sizeof(UnSqueezeParameter)));
  if (unsqueeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnSqueezeParameter failed.";
    return nullptr;
  }
  memset(unsqueeze_param, 0, sizeof(UnSqueezeParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  unsqueeze_param->op_parameter_.type_ = primitive->value_type();
  auto unsqueeze_prim = primitive->value_as_Unsqueeze();
  auto flat_axis = std::vector<int>(unsqueeze_prim->axis()->begin(), unsqueeze_prim->axis()->end());
  unsqueeze_param->num_dim_ = flat_axis.size();
  int i = 0;
  for (auto iter = flat_axis.begin(); iter != flat_axis.end(); ++iter) {
    unsqueeze_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(unsqueeze_param);
}
}  // namespace

Registry g_unsqueezeParameterRegistry(schema::PrimitiveType_Unsqueeze, PopulateUnsqueezeParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
