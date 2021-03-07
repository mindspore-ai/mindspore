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
#include "nnacl/batch_to_space.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateBatchToSpaceParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto batch_to_space_prim = primitive->value_as_BatchToSpace();
  BatchToSpaceParameter *batch_space_param =
    reinterpret_cast<BatchToSpaceParameter *>(malloc(sizeof(BatchToSpaceParameter)));
  if (batch_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchToSpaceParameter failed.";
    return nullptr;
  }
  memset(batch_space_param, 0, sizeof(BatchToSpaceParameter));
  if (primitive->value_type() == schema::v0::PrimitiveType_BatchToSpace) {
    batch_space_param->op_parameter_.type_ = schema::PrimitiveType_BatchToSpace;
  } else {
    batch_space_param->op_parameter_.type_ = schema::PrimitiveType_BatchToSpaceND;
  }

  auto block_shape = batch_to_space_prim->blockShape();
  if (block_shape->size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  auto crops = batch_to_space_prim->crops();
  if (crops->size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << COMM_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    batch_space_param->block_shape_[i] = *(block_shape->begin() + i);
  }

  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    batch_space_param->crops_[i] = *(crops->begin() + i);
  }
  return reinterpret_cast<OpParameter *>(batch_space_param);
}
}  // namespace

Registry g_batchToSpaceV0ParameterRegistry(schema::v0::PrimitiveType_BatchToSpace, PopulateBatchToSpaceParameter,
                                           SCHEMA_V0);
Registry g_batchToSpaceNDV0ParameterRegistry(schema::v0::PrimitiveType_BatchToSpaceND, PopulateBatchToSpaceParameter,
                                             SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
