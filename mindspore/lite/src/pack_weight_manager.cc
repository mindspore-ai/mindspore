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
#include "src/pack_weight_manager.h"
namespace mindspore::lite {
namespace {
#ifndef __ANDROID__
constexpr size_t kMemAlignSize = 64;
#endif
}  // namespace
PackWeightManager *PackWeightManager::GetInstance() {
  static PackWeightManager instance;
  return &instance;
}

STATUS PackWeightManager::InitByBuf(const char *model_buf, size_t model_size, int numa_id) {
#ifdef SHARING_MODEL_WEIGHT
  if (pack_weight_ == nullptr) {
    pack_weight_ = std::make_shared<PackWeight>();
    if (pack_weight_ == nullptr) {
      MS_LOG(ERROR) << "pack_weight_ is nullptr.";
      return RET_ERROR;
    }
  }
  auto status = pack_weight_->InitWeightManagerByBuf(model_buf, model_size, numa_id);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "InitWeightManagerByBuf failed.";
    return RET_ERROR;
  }
#endif
  return RET_OK;
}

char *PackWeightManager::GetNumaModelBuf(const char *model_buf, int numa_id) {
#ifdef SHARING_MODEL_WEIGHT
  return pack_weight_->GetNumaModelBuf(model_buf, numa_id);
#endif
  return nullptr;
}

STATUS PackWeightManager::StoreOriginTensorData(Model *model) {
#ifdef SHARING_MODEL_WEIGHT
  MS_CHECK_TRUE_MSG(model != nullptr, RET_ERROR, "model is nullptr in pack weight manager.");
  if (pack_weight_ == nullptr) {
    MS_LOG(DEBUG) << "define SHARING_MODEL_WEIGHT but not use parallel predict.";
    return RET_OK;
  }
  auto lite_model = reinterpret_cast<LiteModel *>(model);
  auto kernel_num = model->all_nodes_.size();
  for (size_t i = 0; i < kernel_num; i++) {
    auto node = model->all_nodes_[i];
    for (size_t j = 0; j < node->input_indices_.size(); j++) {
      auto tensor_index = node->input_indices_[j];
      auto src_tensor = lite_model->GetSchemaTensor(tensor_index);
      if (src_tensor == nullptr || src_tensor->handler() == nullptr || src_tensor->data() == nullptr ||
          src_tensor->length() == 0) {
        continue;
      }
      auto status = pack_weight_->StoreOriginTensorData(lite_model->buf, src_tensor->data());
      if (status != RET_OK) {
        MS_LOG(DEBUG) << "data not packed.";
        return RET_ERROR;
      }
    }
  }
#endif
  return RET_OK;
}

void *PackWeightManager::MallocData(size_t size) {
  if (size > MAX_MALLOC_SIZE || size == 0) {
    MS_LOG(ERROR) << "malloc size is wrong.";
    return nullptr;
  }
  void *data = nullptr;
#ifdef _WIN32
  size_t round_size = (size + kMemAlignSize - 1) & (~(kMemAlignSize - 1));
  data = _aligned_malloc(round_size, kMemAlignSize);
#elif defined(__ANDROID__)
  data = malloc(size);
#else
  size_t round_size = (size + kMemAlignSize - 1) & (~(kMemAlignSize - 1));
  auto ret = posix_memalign(&data, kMemAlignSize, round_size);
  if (ret != 0) {
    MS_LOG(ERROR) << "posix_memalign failed.";
    return nullptr;
  }
#endif
  return data;
}

void *PackWeightManager::GetPackData(const void *tensor_data, const size_t size, bool *is_packed) {
#ifdef SHARING_MODEL_WEIGHT
  if (pack_weight_ == nullptr) {
    void *data = MallocData(size);
    *is_packed = false;
    return data;
  }
  return pack_weight_->GetPackData(tensor_data, size, is_packed);
#endif
  void *data = MallocData(size);
  *is_packed = false;
  return data;
}

void PackWeightManager::FreeData(void *tensor_data) {
  if (tensor_data != nullptr) {
#ifdef _WIN32
    _aligned_free(tensor_data);
#else
    free(tensor_data);
#endif
    tensor_data = nullptr;
  }
}

void PackWeightManager::Free(void *tensor_data) {
#ifdef SHARING_MODEL_WEIGHT
  if (pack_weight_ == nullptr) {
    FreeData(tensor_data);
  }
  return;
#endif
  FreeData(tensor_data);
}
}  // namespace mindspore::lite
