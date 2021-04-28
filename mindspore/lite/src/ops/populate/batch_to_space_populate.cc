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
#include "nnacl/batch_to_space.h"
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateBatchToSpaceParameter(const void *prim) {
  auto *batch_space_param = reinterpret_cast<BatchToSpaceParameter *>(malloc(sizeof(BatchToSpaceParameter)));
  if (batch_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchToSpaceParameter failed.";
    return nullptr;
  }
  memset(batch_space_param, 0, sizeof(BatchToSpaceParameter));
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  batch_space_param->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_BatchToSpace();
  if (param == nullptr) {
    MS_LOG(ERROR) << "param is nullptr";
    return nullptr;
  }
  auto block_size = param->block_size();
  if (block_size == nullptr) {
    return reinterpret_cast<OpParameter *>(batch_space_param);
  }
  auto block_shape = std::vector<int64_t>(block_size->begin(), block_size->end());
  if (block_shape.size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  auto crop = param->crops();
  if (crop == nullptr) {
    MS_LOG(ERROR) << "crop is nullptr";
    return nullptr;
  }
  auto fb_crops = crop->data();
  if (fb_crops == nullptr) {
    MS_LOG(ERROR) << "fb_crops is nullptr";
    return nullptr;
  }
  std::vector<int64_t> crops;
  for (auto iter = fb_crops->begin(); iter != fb_crops->end(); ++iter) {
    auto crops_data = (*iter)->data();
    if (crops_data == nullptr) {
      MS_LOG(ERROR) << "crops_data is nullptr";
      return nullptr;
    }
    auto crops_vec = std::vector<int64_t>(crops_data->begin(), crops_data->end());
    crops.insert(crops.end(), crops_vec.begin(), crops_vec.end());
  }
  if (crops.size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << COMM_SHAPE_SIZE;
    free(batch_space_param);
    return nullptr;
  }

  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    batch_space_param->block_shape_[i] = static_cast<int>(block_shape[i]);
  }

  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    batch_space_param->crops_[i] = static_cast<int>(crops[i]);
  }
  return reinterpret_cast<OpParameter *>(batch_space_param);
}
}  // namespace
REG_POPULATE(PrimitiveType_BatchToSpace, PopulateBatchToSpaceParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_BatchToSpaceND, PopulateBatchToSpaceParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
