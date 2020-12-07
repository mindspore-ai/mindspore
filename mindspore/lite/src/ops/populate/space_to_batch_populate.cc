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

#include "src/ops/space_to_batch.h"
#include "src/common/common.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/space_to_batch_fp32.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSpaceToBatchParameter(const mindspore::lite::PrimitiveC *primitive) {
  SpaceToBatchParameter *space_batch_param =
    reinterpret_cast<SpaceToBatchParameter *>(malloc(sizeof(SpaceToBatchParameter)));
  if (space_batch_param == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToBatchParameter failed.";
    return nullptr;
  }
  memset(space_batch_param, 0, sizeof(SpaceToBatchParameter));
  space_batch_param->op_parameter_.type_ = primitive->Type();
  auto block_sizes = ((mindspore::lite::SpaceToBatch *)primitive)->BlockSizes();
  space_batch_param->m_ = block_sizes.size();
  if (block_sizes.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of block_sizes.size() is too big";
    free(space_batch_param);
    return nullptr;
  }
  memcpy(space_batch_param->block_sizes_, (block_sizes.data()), block_sizes.size() * sizeof(int));
  auto paddings = ((mindspore::lite::SpaceToBatch *)primitive)->Paddings();
  if (paddings.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of paddings.size() is too big";
    free(space_batch_param);
    return nullptr;
  }
  memcpy(space_batch_param->paddings_, (paddings.data()), paddings.size() * sizeof(int));
  return reinterpret_cast<OpParameter *>(space_batch_param);
}
Registry SpaceToBatchParameterRegistry(schema::PrimitiveType_SpaceToBatch, PopulateSpaceToBatchParameter);

}  // namespace lite
}  // namespace mindspore
