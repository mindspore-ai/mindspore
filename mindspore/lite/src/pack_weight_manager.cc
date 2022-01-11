/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifdef USING_SERVING
#include "src/pack_weight_manager.h"
namespace mindspore::lite {
PackWeightManager *PackWeightManager::GetInstance() {
  static PackWeightManager instance;
  return &instance;
}

void PackWeightManager::InitWeightManagerByBuf(const char *model_buf, const LiteSession *lite_session) {
  MS_CHECK_TRUE_RET_VOID(model_buf != nullptr);
  MS_CHECK_TRUE_RET_VOID(lite_session != nullptr);
  if (buf_model_weight_.find(model_buf) == buf_model_weight_.end()) {
    auto *model_const_weight = new (std::nothrow) ModelConstWeight();
    if (model_const_weight == nullptr) {
      return;
    }
    buf_model_weight_[model_buf] = model_const_weight;
  }
  buf_model_weight_[model_buf]->lite_sessions.push_back(lite_session);
}

void PackWeightManager::InitWeightManagerByPath(const std::string &model_path, const char *model_buf,
                                                const LiteSession *session) {
  MS_CHECK_TRUE_RET_VOID(model_buf != nullptr);
  if (path_model_buf_.find(model_path) == path_model_buf_.end()) {
    auto *model_const_weight = new (std::nothrow) ModelConstWeight();
    if (model_const_weight == nullptr) {
      return;
    }
    path_model_weight_[model_path] = model_const_weight;
  }
  path_model_weight_[model_path]->lite_sessions.push_back(session);
  path_model_buf_[model_path].push_back(model_buf);
}

STATUS PackWeightManager::StoreLiteModel(const char *model_buf, const Model *model) {
  MS_CHECK_TRUE_RET(model_buf != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(model != nullptr, RET_ERROR);
  for (auto &item : path_model_buf_) {
    auto &model_bufs = item.second;
    auto path = item.first;
    if (find(model_bufs.begin(), model_bufs.end(), model_buf) != model_bufs.end()) {
      path_model_weight_[path]->lite_models.push_back(model);
      return RET_OK;
    }
  }
  if (buf_model_weight_.find(model_buf) == buf_model_weight_.end()) {
    MS_LOG(ERROR) << "Set model failed.";
    return RET_ERROR;
  }
  buf_model_weight_[model_buf]->lite_models.push_back(model);
  return RET_OK;
}

void PackWeightManager::StoreOriginTensor(const LiteModel *model, const SchemaTensorWrapper *origin_tensor,
                                          size_t tensor_index) {
  MS_CHECK_TRUE_RET_VOID(model != nullptr);
  MS_CHECK_TRUE_RET_VOID(origin_tensor != nullptr);
  for (auto &item : buf_model_weight_) {
    auto &model_buf = item.first;
    auto &model_weight = item.second;
    for (auto &lite_model : model_weight->lite_models) {
      if (model == lite_model) {
        if (model_weight->origin_weight.find(tensor_index) == model_weight->origin_weight.end()) {
          buf_model_weight_[model_buf]->origin_weight[tensor_index] = origin_tensor->data();
        }
      }
    }
  }
  for (auto &item : path_model_weight_) {
    auto &path = item.first;
    auto &model_weight = item.second;
    for (auto &lite_model : model_weight->lite_models) {
      if (model == lite_model) {
        if (model_weight->origin_weight.find(tensor_index) == model_weight->origin_weight.end()) {
          path_model_weight_[path]->origin_weight[tensor_index] = origin_tensor->data();
        }
      }
    }
  }
}

void *PackWeightManager::GetTensorData(const LiteModel *model, size_t tensor_index) {
  MS_CHECK_TRUE_RET(model != nullptr, nullptr);
  for (auto &item : buf_model_weight_) {
    auto &model_weight = item.second;
    auto &models = model_weight->lite_models;
    if (find(models.begin(), models.end(), model) != models.end()) {
      if (model_weight->packed_weight.find(tensor_index) != model_weight->packed_weight.end()) {
        return model_weight->packed_weight[tensor_index];
      }
    }
  }
  for (auto &item : path_model_weight_) {
    auto &model_weight = item.second;
    auto &models = model_weight->lite_models;
    if (find(models.begin(), models.end(), model) != models.end()) {
      if (model_weight->packed_weight.find(tensor_index) != model_weight->packed_weight.end()) {
        return model_weight->packed_weight[tensor_index];
      }
    }
  }
  return nullptr;
}

std::pair<PackStatus, void *> PackWeightManager::FindPackedTensor(PackedWeight *packed_weights,
                                                                  const OriginWeight &origin_weithts,
                                                                  const Tensor *tensor, const size_t size) {
  MS_CHECK_TRUE_RET(packed_weights != nullptr, std::make_pair(MALLOC, nullptr));
  MS_CHECK_TRUE_RET(tensor != nullptr, std::make_pair(MALLOC, nullptr));
  if (size > MAX_MALLOC_SIZE) {
    MS_LOG(ERROR) << "malloc size more than MAX_MALLOC_SIZE";
    return std::make_pair(MALLOC, nullptr);
  }
  for (auto &packed_weight : *packed_weights) {
    auto &packed_tensor = packed_weight.second;
    if (packed_tensor == tensor->data()) {
      return std::make_pair(PACKED, packed_tensor);
    }
  }
  for (auto &origin_weight : origin_weithts) {
    auto &origin_tensor = origin_weight.second;
    auto &origin_index = origin_weight.first;
    if (origin_tensor == tensor->data()) {
      void *data = malloc(size);
      if (data == nullptr) {
        MS_LOG(ERROR) << "malloc failed.";
        return std::make_pair(MALLOC, nullptr);
      }
      memset(data, 0, size);
      packed_weights->insert(std::make_pair(origin_index, data));
      return std::make_pair(NOTPACK, packed_weights->at(origin_index));
    }
  }
  return std::make_pair(MALLOC, nullptr);
}

std::pair<PackStatus, void *> PackWeightManager::GetPackedTensor(const Tensor *tensor, const size_t size) {
  MS_CHECK_TRUE_RET(tensor != nullptr, std::make_pair(MALLOC, nullptr));
  std::pair<PackStatus, void *> packed_tensor_pair;
  for (auto &item : buf_model_weight_) {
    auto &model_weight = item.second;
    auto &origin_weithts = model_weight->origin_weight;
    auto &packed_weights = model_weight->packed_weight;
    packed_tensor_pair = FindPackedTensor(&packed_weights, origin_weithts, tensor, size);
    if (packed_tensor_pair.second != nullptr) {
      return packed_tensor_pair;
    }
  }

  for (auto &item : path_model_weight_) {
    auto &model_weight = item.second;
    auto &origin_weithts = model_weight->origin_weight;
    auto &packed_weights = model_weight->packed_weight;
    packed_tensor_pair = FindPackedTensor(&packed_weights, origin_weithts, tensor, size);
    if (packed_tensor_pair.second != nullptr) {
      return packed_tensor_pair;
    }
  }
  return std::make_pair(MALLOC, nullptr);
}

void PackWeightManager::DeleteSavedModelPtr(LiteModel *delete_model) {
  MS_CHECK_TRUE_RET_VOID(delete_model != nullptr);
  for (auto &item : path_model_weight_) {
    auto &weight = item.second;
    auto it = find(weight->lite_models.begin(), weight->lite_models.end(), delete_model);
    if (it != weight->lite_models.end()) {
      weight->lite_models.erase(it);
    }
  }
  for (auto &item : buf_model_weight_) {
    auto &weight = item.second;
    auto it = find(weight->lite_models.begin(), weight->lite_models.end(), delete_model);
    if (it != weight->lite_models.end()) {
      weight->lite_models.erase(it);
    }
  }
}

void PackWeightManager::DeleteSavedSessionPtr(LiteSession *delete_session) {
  MS_CHECK_TRUE_RET_VOID(delete_session != nullptr);
  for (auto &item : path_model_weight_) {
    auto &weight = item.second;
    auto it = find(weight->lite_sessions.begin(), weight->lite_sessions.end(), delete_session);
    if (it != weight->lite_sessions.end()) {
      weight->lite_sessions.erase(it);
    }
  }
  for (auto &item : buf_model_weight_) {
    auto &weight = item.second;
    auto it = find(weight->lite_sessions.begin(), weight->lite_sessions.end(), delete_session);
    if (it != weight->lite_sessions.end()) {
      weight->lite_sessions.erase(it);
    }
  }
}

void PackWeightManager::FreePackedWeight(ModelConstWeight *weight) {
  auto &packed_tensors = weight->packed_weight;
  for (auto &packed_tensor : packed_tensors) {
    if (packed_tensor.second != nullptr) {
      free(packed_tensor.second);
      packed_tensor.second = nullptr;
    }
  }
  if (weight != nullptr) {
    delete weight;
    weight = nullptr;
  }
}

void PackWeightManager::FreeBufModelWeight() {
  for (auto &item : buf_model_weight_) {
    FreePackedWeight(item.second);
  }
}

void PackWeightManager::FreePathModelWeight() {
  for (auto &item : path_model_weight_) {
    FreePackedWeight(item.second);
  }
}

PackWeightManager::~PackWeightManager() {
  FreePathModelWeight();
  FreeBufModelWeight();
}
}  // namespace mindspore::lite
#endif
