/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "src/runtime/pack_weight.h"
#include "src/runtime/dynamic_mem_allocator.h"
namespace mindspore::lite {
STATUS PackWeight::InitWeightManagerByBuf(const char *model_buf, size_t model_size, int numa_id) {
  MS_CHECK_TRUE_MSG(model_buf != nullptr, RET_ERROR, "model buf is nullptr in pack weight manager.");
  if (model_buf_map_.find(model_buf) != model_buf_map_.end() &&
      find(numa_model_buf_[model_buf].begin(), numa_model_buf_[model_buf].end(), numa_id) !=
        numa_model_buf_[model_buf].end()) {
    MS_LOG(DEBUG) << "same numa id, use same model buf.";
    return RET_OK;
  }
  // model buf and weight use same allocator, create in weight pack manager
  auto allocator = std::make_shared<DynamicMemAllocator>(numa_id);
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "allocator is nullptr in pack weight manager.";
    return RET_ERROR;
  }
  auto new_model_buf = static_cast<char *>(allocator->Malloc(model_size));
  if (new_model_buf == nullptr) {
    MS_LOG(ERROR) << "new model buf is nullptr in pack weight manager.";
    return RET_ERROR;
  }
  memcpy(new_model_buf, model_buf, model_size);
  numa_model_buf_[model_buf] = {numa_id};
  auto *model_const_weight = new (std::nothrow) ModelConstWeight();
  if (model_const_weight == nullptr) {
    MS_LOG(ERROR) << "model const weight is nullptr.";
    return RET_ERROR;
  }
  model_const_weight->numa_id = numa_id;
  buf_model_weight_[new_model_buf] = model_const_weight;
  buf_model_weight_[new_model_buf]->allocator = allocator;
  model_buf_map_.insert(std::make_pair(model_buf, new_model_buf));
  return RET_OK;
}

char *PackWeight::GetNumaModelBuf(const char *model_buf, int numa_id) {
  if (model_buf_map_.find(model_buf) == model_buf_map_.end() ||
      find(numa_model_buf_[model_buf].begin(), numa_model_buf_[model_buf].end(), numa_id) ==
        numa_model_buf_[model_buf].end()) {
    MS_LOG(ERROR) << "can not find numa id in saved model buf.";
    return nullptr;
  }
  return model_buf_map_[model_buf];
}

STATUS PackWeight::StoreOriginTensorData(const char *model_buf, const void *origin_tensor_data) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (buf_model_weight_.find(model_buf) == buf_model_weight_.end()) {
    MS_LOG(ERROR) << "can not find model buf in store origin Tensor";
    return RET_ERROR;
  }
  auto &model_weight = buf_model_weight_[model_buf];
  auto &packed_pair = model_weight->origin_and_packed_pair;
  if (packed_pair.find(origin_tensor_data) != packed_pair.end()) {
    MS_LOG(DEBUG) << "origin tensor data already store by other model.";
    return RET_OK;
  }
  packed_pair.insert(std::make_pair(origin_tensor_data, nullptr));
  return RET_OK;
}

void *PackWeight::GetPackData(const void *tensor_data, const size_t size, bool *is_packed) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  MS_CHECK_TRUE_RET(tensor_data != nullptr, nullptr);
  for (auto &item : buf_model_weight_) {
    auto &model_weight = item.second;
    auto &origin_packed_weight = model_weight->origin_and_packed_pair;
    if (origin_packed_weight.find(tensor_data) == origin_packed_weight.end()) {
      continue;
    }
    auto packed_tensor_data = origin_packed_weight[tensor_data];
    if (packed_tensor_data != nullptr) {
      *is_packed = true;
      return packed_tensor_data;
    } else {
      auto weight_allocator = model_weight->allocator;
      packed_tensor_data = weight_allocator->Malloc(size);
      origin_packed_weight[tensor_data] = packed_tensor_data;
      *is_packed = false;
      return packed_tensor_data;
    }
  }
  *is_packed = false;
  MS_LOG(ERROR) << "can not find tensor data in origin tensor data.";
  return nullptr;
}

void PackWeight::FreePackedWeight(ModelConstWeight *weight) {
  MS_CHECK_TRUE_RET_VOID(weight != nullptr);
  for (auto &origin_and_packed_pair : weight->origin_and_packed_pair) {
    auto &packed_data = origin_and_packed_pair.second;
    auto allocator = weight->allocator;
    if (packed_data != nullptr) {
      allocator->Free(packed_data);
      packed_data = nullptr;
    }
  }
  weight->origin_and_packed_pair.clear();
}

PackWeight::~PackWeight() {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  for (auto &item : buf_model_weight_) {
    FreePackedWeight(item.second);
  }
  // free model buf
  for (auto &item : buf_model_weight_) {
    auto model_buf = const_cast<char *>(item.first);
    auto &allocator = item.second->allocator;
    allocator->Free(model_buf);
    if (item.second != nullptr) {
      delete item.second;
      item.second = nullptr;
    }
  }
  buf_model_weight_.clear();
}
}  // namespace mindspore::lite
