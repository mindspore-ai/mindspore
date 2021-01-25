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
#include "nnacl/fp32/topk_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateTopKParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto topk_prim = primitive->value_as_TopK();
  TopkParameter *topk_param = reinterpret_cast<TopkParameter *>(malloc(sizeof(TopkParameter)));
  if (topk_param == nullptr) {
    MS_LOG(ERROR) << "malloc TopkParameter failed.";
    return nullptr;
  }
  memset(topk_param, 0, sizeof(TopkParameter));
  topk_param->op_parameter_.type_ = schema::PrimitiveType_TopKFusion;

  topk_param->k_ = topk_prim->k();
  topk_param->sorted_ = topk_prim->sorted();
  return reinterpret_cast<OpParameter *>(topk_param);
}
}  // namespace

Registry g_topKV0ParameterRegistry(schema::v0::PrimitiveType_TopK, PopulateTopKParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
