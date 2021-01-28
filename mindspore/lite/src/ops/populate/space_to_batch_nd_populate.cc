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

#include "src/ops/space_to_batch_nd.h"
#include "src/common/common.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/space_to_batch_fp32.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateSpaceToBatchNDParameter(const mindspore::lite::PrimitiveC *primitive) {
  auto *space_batch_param_nd = reinterpret_cast<SpaceToBatchParameter *>(malloc(sizeof(SpaceToBatchParameter)));
  if (space_batch_param_nd == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToBatchParameter failed.";
    return nullptr;
  }

  space_batch_param_nd->op_parameter_.type_ = primitive->Type();
  auto block_sizes = ((mindspore::lite::SpaceToBatchND *)primitive)->GetBlockShape();
  if (block_sizes.empty()) {
    return reinterpret_cast<OpParameter *>(space_batch_param_nd);
  }
  space_batch_param_nd->m_ = block_sizes.size();
  if (block_sizes.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of block_sizes.size() is too big";
    free(space_batch_param_nd);
    return nullptr;
  }
  memcpy(space_batch_param_nd->block_sizes_, (block_sizes.data()), block_sizes.size() * sizeof(int));
  auto paddings = ((mindspore::lite::SpaceToBatchND *)primitive)->GetPaddings();
  if (paddings.empty()) {
    return reinterpret_cast<OpParameter *>(space_batch_param_nd);
  }
  if (paddings.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of paddings.size() is too big";
    free(space_batch_param_nd);
    return nullptr;
  }
  memcpy(space_batch_param_nd->paddings_, (paddings.data()), paddings.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(space_batch_param_nd);
}
Registry SpaceToBatchNDParameterRegistry(schema::PrimitiveType_SpaceToBatchND, PopulateSpaceToBatchNDParameter);

}  // namespace lite
}  // namespace mindspore
