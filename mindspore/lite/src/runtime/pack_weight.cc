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
STATUS PackWeight::InitWeightManagerByBuf(const char *model_buf, size_t model_size, int numa_id, bool copy_buf) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  MS_CHECK_TRUE_MSG(model_buf != nullptr, RET_ERROR, "model buf is nullptr in pack weight manager.");
  copy_buf_ = copy_buf;
  if (model_buf_map_.find(model_buf) != model_buf_map_.end() &&
      find(numa_model_buf_[model_buf].begin(), numa_model_buf_[model_buf].end(), numa_id) !=
        numa_model_buf_[model_buf].end()) {
    MS_LOG(DEBUG) << "same numa id, use same model buf.";
    return RET_OK;
  }
  // model buf and weight use same allocator, create in weight pack manager
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
  const char *new_model_buf;
  if (copy_buf_) {
    auto numa_model_buf = static_cast<char *>(allocator->Malloc(model_size));
    if (numa_model_buf == nullptr) {
      MS_LOG(ERROR) << "new model buf is nullptr in pack weight manager.";
      return RET_ERROR;
    }
    memcpy(numa_model_buf, model_buf, model_size);
    new_model_buf = numa_model_buf;
  } else {
    new_model_buf = model_buf;
  }
  if (numa_model_buf_.find(model_buf) == numa_model_buf_.end()) {
    numa_model_buf_[model_buf] = {numa_id};
    model_buf_map_[model_buf] = {new_model_buf};
  } else {
    numa_model_buf_[model_buf].push_back(numa_id);
    model_buf_map_[model_buf].push_back(new_model_buf);
  }
  buf_model_weight_[new_model_buf] = model_const_weight;
  buf_model_weight_[new_model_buf]->allocator = allocator;
  model_const_weight->numa_id = numa_id;
  return RET_OK;
}

const char *PackWeight::GetNumaModelBuf(const char *model_buf, int numa_id) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (model_buf_map_.find(model_buf) == model_buf_map_.end() ||
      find(numa_model_buf_[model_buf].begin(), numa_model_buf_[model_buf].end(), numa_id) ==
        numa_model_buf_[model_buf].end()) {
    MS_LOG(ERROR) << "can not find numa id in saved model buf.";
    return nullptr;
  }
  auto numa_id_list = numa_model_buf_[model_buf];
  auto it = find(numa_id_list.begin(), numa_id_list.end(), numa_id) - numa_id_list.begin();
  return model_buf_map_[model_buf][it];
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

void *PackWeight::ReplaceFp16Data(void *origin_fp16_data, size_t size) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (fp16_fp32_data_pair_.find(origin_fp16_data) != fp16_fp32_data_pair_.end()) {
    return fp16_fp32_data_pair_[origin_fp16_data];
  } else {
    for (auto &item : buf_model_weight_) {
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
  MS_LOG(ERROR) << "ReplaceFp16Data failed.";
  return nullptr;
}

STATUS PackWeight::ReplaceOriginTensorData(const char *model_buf, std::vector<Tensor *> *tensors, int tensor_index) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  if (buf_model_weight_.find(model_buf) == buf_model_weight_.end()) {
    MS_LOG(ERROR) << "can not find model buf in store origin Tensor";
    return RET_ERROR;
  }
  auto &tensor = tensors->at(tensor_index);
  auto &model_weight = buf_model_weight_[model_buf];
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
      if (packed_tensor_data == nullptr) {
        MS_LOG(ERROR) << "malloc failed.";
        return nullptr;
      }
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

void PackWeight::DeleteOriginModelBufInfo(const char *model_buf) {
  std::lock_guard<std::mutex> lock(mtx_weight_);
  numa_model_buf_.erase(model_buf);
  model_buf_map_.erase(model_buf);
}

void PackWeight::FreePackWeight(std::vector<const char *> model_bufs, bool all) {
  MS_LOG(INFO) << "free pack weight by other model buf.";
  std::lock_guard<std::mutex> lock(mtx_weight_);
  for (auto &item : buf_model_weight_) {
    auto model_buf = const_cast<char *>(item.first);
    if (!all && find(model_bufs.begin(), model_bufs.end(), model_buf) == model_bufs.end()) {
      continue;
    }
    FreePackedWeight(item.second);
    FreeFp16ToFp32Data(item.second);
    FreeTensorData(item.second);
  }
  // free model buf
  if (copy_buf_) {
    for (auto &item : buf_model_weight_) {
      auto model_buf = const_cast<char *>(item.first);
      if (!all && find(model_bufs.begin(), model_bufs.end(), model_buf) == model_bufs.end()) {
        continue;
      }
      if (item.second != nullptr) {
        auto &allocator = item.second->allocator;
        allocator->Free(model_buf);
        delete item.second;
        item.second = nullptr;
      }
    }
  } else {
    for (auto &item : buf_model_weight_) {
      if (item.second != nullptr) {
        delete item.second;
        item.second = nullptr;
      }
    }
  }

  for (auto &buf : model_bufs) {
    buf_model_weight_.erase(buf);
  }
  if (all) {
    buf_model_weight_.clear();
    numa_model_buf_.clear();
    model_buf_map_.clear();
  }
}

PackWeight::~PackWeight() {
  MS_LOG(INFO) << "free pack weight.";
  FreePackWeight({}, true);
  MS_LOG(INFO) << "free pack weight done.";
}
}  // namespace mindspore::lite
