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

#include "src/litert/pack_weight.h"
#include "src/extendrt/dynamic_mem_allocator.h"
namespace mindspore::lite {
STATUS PackWeight::InitPackWeight(const void *model_buf, size_t model_size, std::string id, int numa_id,
                                  bool need_copy_buf) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (model_buf == nullptr || model_weights_.size() != shared_bufs_.size()) {
    MS_LOG(ERROR) << "model buf is nullptr in pack weight manager.";
    return RET_ERROR;
  }
  if (model_weights_.find(id) != model_weights_.end() && model_weights_[id].find(numa_id) != model_weights_[id].end()) {
    MS_LOG(INFO) << "same numa id, use same model buf.";
    return RET_OK;
  }
  std::shared_ptr<Allocator> allocator = nullptr;
#ifdef BFC_MEMORY
  allocator = std::make_shared<DynamicMemAllocator>(numa_id);
#else
  allocator = std::make_shared<DefaultAllocator>();
#endif
  if (allocator == nullptr) {
    MS_LOG(ERROR) << "allocator is nullptr in pack weight manager.";
    return RET_ERROR;
  }
  auto *model_const_weight = new (std::nothrow) ModelConstWeight();
  if (model_const_weight == nullptr) {
    MS_LOG(ERROR) << "model const weight is nullptr.";
    return RET_ERROR;
  }
  void *new_model_buf = const_cast<void *>(model_buf);
  if (need_copy_buf) {
    new_model_buf = allocator->Malloc(model_size);
    if (new_model_buf == nullptr) {
      MS_LOG(ERROR) << "new model buf is nullptr in pack weight manager.";
      return RET_ERROR;
    }
    memcpy(new_model_buf, model_buf, model_size);
    model_const_weight->copy_buf = need_copy_buf;
  }
  model_const_weight->allocator = allocator;
  model_const_weight->numa_id = numa_id;
  if (model_weights_.find(id) != model_weights_.end()) {
    model_weights_[id][numa_id] = model_const_weight;
    shared_bufs_[id][numa_id] = new_model_buf;
  } else {
    std::unordered_map<int, ModelConstWeight *> numa_model_weight;
    numa_model_weight[numa_id] = model_const_weight;
    model_weights_[id] = numa_model_weight;
    std::unordered_map<int, void *> numa_model_buf;
    numa_model_buf[numa_id] = new_model_buf;
    shared_bufs_[id] = numa_model_buf;
  }
  return RET_OK;
}

char *PackWeight::GetSharedModelBuf(std::string id, int numa_id) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (shared_bufs_.find(id) == shared_bufs_.end() || shared_bufs_[id].find(numa_id) == shared_bufs_[id].end()) {
    MS_LOG(ERROR) << "can not find numa id in saved model buf, id: " << id << ", numa id: " << numa_id;
    return nullptr;
  }
  return static_cast<char *>(shared_bufs_[id][numa_id]);
}

STATUS PackWeight::StoreOriginTensorData(const void *model_buf, const void *origin_tensor_data) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  for (auto &item : shared_bufs_) {
    for (auto &numa_item : item.second) {
      if (numa_item.second == model_buf) {
        std::string id = item.first;
        int numa_id = numa_item.first;
        auto &model_weight = model_weights_[id][numa_id];
        auto &packed_pair = model_weight->origin_and_packed_pair;
        if (packed_pair.find(origin_tensor_data) != packed_pair.end()) {
          MS_LOG(DEBUG) << "origin tensor data already store by other model.";
          return RET_OK;
        }
        packed_pair.insert(std::make_pair(origin_tensor_data, nullptr));
        return RET_OK;
      }
    }
  }
  MS_LOG(ERROR) << "can not find model buf in store origin Tensor";
  return RET_ERROR;
}

void *PackWeight::ReplaceFp16Data(void *origin_fp16_data, size_t size) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (fp16_fp32_data_pair_.find(origin_fp16_data) != fp16_fp32_data_pair_.end()) {
    return fp16_fp32_data_pair_[origin_fp16_data];
  } else {
    for (auto &numa_item : model_weights_) {
      for (auto &item : numa_item.second) {
        if (item.second->origin_and_packed_pair.find(origin_fp16_data) != item.second->origin_and_packed_pair.end()) {
          auto &model_weight = item.second;
          auto &origin_and_packed_pair = model_weight->origin_and_packed_pair;
          if (origin_and_packed_pair.find(origin_fp16_data) == origin_and_packed_pair.end()) {
            MS_LOG(ERROR) << "origin fp16 data not find.";
            return nullptr;
          }
          auto allocator = model_weight->allocator;
          void *data = allocator->Malloc(size);
          if (data == nullptr) {
            MS_LOG(ERROR) << "malloc failed.";
            return nullptr;
          }
          origin_and_packed_pair.insert(std::make_pair(data, nullptr));
          model_weight->fp16_fp32_data.insert(data);
          origin_and_packed_pair.erase(origin_fp16_data);
          fp16_fp32_data_pair_.insert(std::make_pair(origin_fp16_data, data));
          return data;
        }
      }
    }
  }
  MS_LOG(ERROR) << "ReplaceFp16Data failed.";
  return nullptr;
}

STATUS PackWeight::ReplaceOriginTensorData(const void *model_buf, std::vector<Tensor *> *tensors, int tensor_index) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  for (auto &item : shared_bufs_) {
    for (auto &numa_item : item.second) {
      if (numa_item.second == model_buf) {
        std::string id = item.first;
        int numa_id = numa_item.first;
        auto &tensor = tensors->at(tensor_index);
        auto &model_weight = model_weights_[id][numa_id];
        if (model_weight->tensors_data.find(tensor_index) == model_weight->tensors_data.end()) {
          auto allocator = model_weight->allocator;
          void *new_data = allocator->Malloc(tensor->Size());
          if (new_data == nullptr) {
            MS_LOG(ERROR) << "allocator malloc data failed.";
            return RET_ERROR;
          }
          memcpy(new_data, tensor->data(), tensor->Size());
          MS_CHECK_TRUE_MSG(tensor->own_data(), RET_ERROR, "tensor data is not own data.");
          tensor->FreeData();
          tensor->set_data(new_data);
          tensor->set_own_data(false);
          model_weight->tensors_data.insert(std::make_pair(tensor_index, new_data));
        } else {
          auto new_data = model_weight->tensors_data[tensor_index];
          tensor->FreeData();
          tensor->set_data(new_data);
          tensor->set_own_data(false);
        }
        return RET_OK;
      }
    }
  }
  MS_LOG(ERROR) << "can not find model buf in store origin Tensor";
  return RET_ERROR;
}

void *PackWeight::GetPackData(const void *tensor_data, const size_t size, bool *is_packed) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  MS_CHECK_TRUE_RET(tensor_data != nullptr, nullptr);
  for (auto &numa_item : model_weights_) {
    for (auto &item : numa_item.second) {
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
        if (packed_tensor_data == nullptr) {
          MS_LOG(ERROR) << "malloc failed.";
          return nullptr;
        }
        origin_packed_weight[tensor_data] = packed_tensor_data;
        *is_packed = false;
        return packed_tensor_data;
      }
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
    MS_CHECK_TRUE_RET_VOID(allocator != nullptr);
    if (packed_data != nullptr) {
      allocator->Free(packed_data);
      packed_data = nullptr;
    }
  }
  weight->origin_and_packed_pair.clear();
}

void PackWeight::FreeTensorData(ModelConstWeight *weight) {
  MS_CHECK_TRUE_RET_VOID(weight != nullptr);
  for (auto &tensor_data : weight->tensors_data) {
    auto &data = tensor_data.second;
    auto allocator = weight->allocator;
    MS_CHECK_TRUE_RET_VOID(allocator != nullptr);
    if (data != nullptr) {
      allocator->Free(data);
      data = nullptr;
    }
  }
  weight->tensors_data.clear();
}

void PackWeight::FreeFp16ToFp32Data(ModelConstWeight *weight) {
  MS_CHECK_TRUE_RET_VOID(weight != nullptr);
  for (auto &data : weight->fp16_fp32_data) {
    auto allocator = weight->allocator;
    MS_CHECK_TRUE_RET_VOID(allocator != nullptr);
    if (data != nullptr) {
      allocator->Free(data);
    }
  }
  weight->fp16_fp32_data.clear();
}

void PackWeight::FreePackWeight(std::string id, bool free_all) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  MS_LOG(INFO) << "model weight size: " << model_weights_.size() << " | shared buf size: " << shared_bufs_.size();
  if (model_weights_.find(id) == model_weights_.end() || shared_bufs_.find(id) == shared_bufs_.end()) {
    MS_LOG(INFO) << "can not find id in shared bufs or model weights.";
    return;
  }
  for (auto &item : model_weights_[id]) {
    auto numa_id = item.first;
    ModelConstWeight *model_weight = model_weights_[id][numa_id];
    void *model_buf = shared_bufs_[id][numa_id];
    if (model_buf == nullptr || model_weight == nullptr) {
      MS_LOG(ERROR) << "model buf or model weight is nullptr.";
      return;
    }
    FreePackedWeight(model_weight);
    FreeFp16ToFp32Data(model_weight);
    FreeTensorData(model_weight);
    if (model_weight->copy_buf) {
      auto &allocator = model_weight->allocator;
      allocator->Free(model_buf);
      model_buf = nullptr;
    }
    delete model_weight;
    model_weight = nullptr;
  }
  if (!free_all) {
    model_weights_.erase(id);
  }
  shared_bufs_.erase(id);
  MS_LOG(INFO) << "FreePackWeight done.";
}

PackWeight::~PackWeight() {
  MS_LOG(INFO) << "~PackWeight() begin";
  if (model_weights_.empty()) {
    MS_LOG(INFO) << "~PackWeight() empty end";
    return;
  }
  for (auto &numa_item : model_weights_) {
    std::string id = numa_item.first;
    FreePackWeight(id, true);
  }
  model_weights_.clear();
  MS_LOG(INFO) << "~PackWeight() end";
}
}  // namespace mindspore::lite
