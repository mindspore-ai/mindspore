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

#include "src/ops/gather.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/gather_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateGatherParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto gather_attr = reinterpret_cast<mindspore::lite::Gather *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  GatherParameter *gather_param = reinterpret_cast<GatherParameter *>(malloc(sizeof(GatherParameter)));
  if (gather_param == nullptr) {
    MS_LOG(ERROR) << "malloc GatherParameter failed.";
    return nullptr;
  }
  memset(gather_param, 0, sizeof(GatherParameter));
  gather_param->op_parameter_.type_ = primitive->Type();
  if (gather_attr->GetAxis() < 0) {
    MS_LOG(ERROR) << "axis should be >= 0.";
    free(gather_param);
    return nullptr;
  }
  gather_param->axis_ = gather_attr->GetAxis();
  gather_param->batchDims_ = gather_attr->GetBatchDims();
  return reinterpret_cast<OpParameter *>(gather_param);
}
Registry GatherParameterRegistry(schema::PrimitiveType_Gather, PopulateGatherParameter);

}  // namespace lite
}  // namespace mindspore
