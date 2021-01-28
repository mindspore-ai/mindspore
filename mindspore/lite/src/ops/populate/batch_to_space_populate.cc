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

#include "src/ops/batch_to_space.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/batch_to_space.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateBatchToSpaceParameter(const mindspore::lite::PrimitiveC *primitive) {
  BatchToSpaceParameter *batch_space_param =
    reinterpret_cast<BatchToSpaceParameter *>(malloc(sizeof(BatchToSpaceParameter)));
  if (batch_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchToSpaceParameter failed.";
    return nullptr;
  }
  memset(batch_space_param, 0, sizeof(BatchToSpaceParameter));
  batch_space_param->op_parameter_.type_ = primitive->Type();
  auto param = reinterpret_cast<mindspore::lite::BatchToSpace *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  auto block_shape = param->GetBlockShape();
  if (block_shape.size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  auto crops = param->GetCrops();
  if (crops.size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << COMM_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    batch_space_param->block_shape_[i] = block_shape[i];
  }

  batch_space_param->no_crop_ = true;
  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    batch_space_param->crops_[i] = crops[i];
    if (batch_space_param->crops_[i] != 0) {
      batch_space_param->no_crop_ = false;
    }
  }
  return reinterpret_cast<OpParameter *>(batch_space_param);
}
Registry BatchToSpaceParameterRegistry(schema::PrimitiveType_BatchToSpace, PopulateBatchToSpaceParameter);
Registry BatchToSpaceNDParameterRegistry(schema::PrimitiveType_BatchToSpaceND, PopulateBatchToSpaceParameter);
}  // namespace lite
}  // namespace mindspore
