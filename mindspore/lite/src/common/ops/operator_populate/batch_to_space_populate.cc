/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/batch_to_space.h"
#include "ops/batch_to_space.h"
#include "ops/batch_to_space_nd.h"
using mindspore::ops::kBlockSize;
using mindspore::ops::kCrops;
using mindspore::ops::kNameBatchToSpace;
using mindspore::ops::kNameBatchToSpaceND;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;
namespace mindspore {
namespace lite {
OpParameter *PopulateBatchToSpaceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<BatchToSpaceParameter *>(PopulateOpParameter<BatchToSpaceParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new BatchToSpaceParameter failed.";
    return nullptr;
  }

  auto attr_block_size = base_operator->GetPrim()->GetAttr(kBlockSize);
  if (attr_block_size == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kBlockSize << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto block_size = GetValue<std::vector<int64_t>>(attr_block_size);
  if (block_size.empty()) {
    return reinterpret_cast<OpParameter *>(param);
  }

  auto block_shape = std::vector<int64_t>(block_size.begin(), block_size.end());
  if (block_shape.size() != BATCH_TO_SPACE_BLOCK_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space blockShape size should be " << BATCH_TO_SPACE_BLOCK_SHAPE_SIZE;
    free(param);
    return nullptr;
  }

  auto attr_crop = base_operator->GetPrim()->GetAttr(kCrops);
  if (attr_crop == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kCrops << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto crop = GetValue<std::vector<std::vector<int64_t>>>(attr_crop);
  auto fb_crops = crop.data();
  if (fb_crops == nullptr) {
    MS_LOG(ERROR) << "fb_crops is nullptr";
    free(param);
    return nullptr;
  }
  std::vector<int64_t> crops;
  for (size_t i = 0; i < fb_crops->size(); ++i) {
    auto crops_data = fb_crops[i];
    auto crops_vec = std::vector<int64_t>(crops_data.begin(), crops_data.end());
    crops.insert(crops.end(), crops_vec.begin(), crops_vec.end());
  }
  if (crops.size() != COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "batch_to_space crops size should be " << COMM_SHAPE_SIZE;
    free(param);
    return nullptr;
  }

  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    param->block_shape_[i] = static_cast<int>(block_shape[i]);
  }

  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    param->crops_[i] = static_cast<int>(crops[i]);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameBatchToSpace, PrimitiveType_BatchToSpace, PopulateBatchToSpaceOpParameter)
REG_OPERATOR_POPULATE(kNameBatchToSpaceND, PrimitiveType_BatchToSpaceND, PopulateBatchToSpaceOpParameter)
}  // namespace lite
}  // namespace mindspore
