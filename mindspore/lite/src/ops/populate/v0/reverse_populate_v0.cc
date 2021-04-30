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
#include "nnacl/fp32/reverse_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateReverseParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto reverse_prim = primitive->value_as_Reverse();
  if (reverse_prim == nullptr) {
    MS_LOG(ERROR) << "reverse_prim is nullptr";
    return nullptr;
  }
  auto *reverse_param = reinterpret_cast<ReverseParameter *>(malloc(sizeof(ReverseParameter)));
  if (reverse_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReverseParameter failed.";
    return nullptr;
  }
  memset(reverse_param, 0, sizeof(ReverseParameter));
  reverse_param->op_parameter_.type_ = schema::PrimitiveType_ReverseV2;
  auto flatAxis = reverse_prim->axis();
  if (flatAxis == nullptr) {
    MS_LOG(ERROR) << "flatAxis is nullptr";
    free(reverse_param);
    return nullptr;
  }
  reverse_param->num_axis_ = flatAxis->size();
  int i = 0;
  for (int flatAxi : *flatAxis) {
    reverse_param->axis_[i++] = flatAxi;
  }
  return reinterpret_cast<OpParameter *>(reverse_param);
}
}  // namespace

Registry g_reverseV0ParameterRegistry(schema::v0::PrimitiveType_Reverse, PopulateReverseParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
