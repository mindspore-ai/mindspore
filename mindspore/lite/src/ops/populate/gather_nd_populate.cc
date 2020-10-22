/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/gather_nd.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/gatherNd.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateGatherNdParameter(const mindspore::lite::PrimitiveC *primitive) {
  GatherNdParameter *gather_nd_param = reinterpret_cast<GatherNdParameter *>(malloc(sizeof(GatherNdParameter)));
  if (gather_nd_param == nullptr) {
    MS_LOG(ERROR) << "malloc GatherNdParameter failed.";
    return nullptr;
  }
  memset(gather_nd_param, 0, sizeof(GatherNdParameter));
  gather_nd_param->op_parameter_.type_ = primitive->Type();
  auto gatherNd_attr =
    reinterpret_cast<mindspore::lite::GatherNd *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  gather_nd_param->batchDims_ = gatherNd_attr->GetBatchDims();
  return reinterpret_cast<OpParameter *>(gather_nd_param);
}

Registry GatherNdParameterRegistry(schema::PrimitiveType_GatherNd, PopulateGatherNdParameter);

}  // namespace lite
}  // namespace mindspore
