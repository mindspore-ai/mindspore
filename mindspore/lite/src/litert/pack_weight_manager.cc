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
#include "src/litert/pack_weight_manager.h"
#include <vector>
#include <map>
#include <string>
#include "src/common/graph_util.h"
namespace mindspore::lite {
namespace {
#ifndef __ANDROID__
constexpr size_t kMemAlignSize = 64;
#endif

#ifdef SHARING_MODEL_WEIGHT
std::string ParseNumaId(const std::map<std::string, std::map<std::string, std::string>> *config_info) {
  std::string numa_id = "-1";
  if (config_info == nullptr) {
    return numa_id;
  }
  auto it_id = config_info->find(kInnerIDs);
  if (it_id != config_info->end()) {
    auto item_numa = it_id->second.find(kInnerNumaID);
    if (item_numa != it_id->second.end()) {
      numa_id = it_id->second.at(kInnerNumaID);
    }
  }
  return numa_id;
}

std::string ParseRunnerId(const std::map<std::string, std::map<std::string, std::string>> *config_info) {
  std::string runner_id;
  if (config_info == nullptr) {
    return runner_id;
  }
  auto it_id = config_info->find(kInnerIDs);
  if (it_id != config_info->end()) {
    auto item_runner = it_id->second.find(kInnerRunnerID);
    if (item_runner != it_id->second.end()) {
      runner_id = it_id->second.at(kInnerRunnerID);
    }
  }
  return runner_id;
}
#endif
}  // namespace
PackWeightManager *PackWeightManager::GetInstance() {
  static PackWeightManager instance;
  return &instance;
}

std::string PackWeightManager::GenRunnerID() {
  std::unique_lock<std::mutex> l(manager_mutex_);
  std::string runner_id = "runner_" + std::to_string(runner_id_);
  runner_ids_.push_back(runner_id);
  runner_id_++;
  MS_LOG(INFO) << "generate runner id: " << runner_id;
  return runner_id;
}

std::string PackWeightManager::GenModelID() {
  std::string model_id = "model_" + std::to_string(model_id_);
  model_ids_.push_back(model_id);
  model_id_++;
  MS_LOG(INFO) << "generate model id: " << model_id;
  return model_id;
}

bool PackWeightManager::IsCopyTensor(int op_type) {
#ifdef SHARING_MODEL_WEIGHT
  return true;
#endif
  if (IsPackedOp(op_type)) {
    return true;
  }
  return false;
}

STATUS PackWeightManager::InitPackWeightManager(
  const char *model_buf, size_t model_size, std::string *model_id,
  const std::map<std::string, std::map<std::string, std::string>> *config_info) {
#ifdef SHARING_MODEL_WEIGHT
  std::unique_lock<std::mutex> l(manager_mutex_);
  if (pack_weight_ == nullptr) {
    pack_weight_ = std::make_shared<PackWeight>();
    if (pack_weight_ == nullptr) {
      MS_LOG(ERROR) << "pack_weight_ is nullptr.";
      return RET_ERROR;
    }
  }
  auto numa_id = std::atoi(ParseNumaId(config_info).c_str());
  *model_id = GenModelID();
  std::string id = ParseRunnerId(config_info);
  if (id.empty()) {
    MS_LOG(INFO) << "model use share pack weight.";
    id = *model_id;
  }
  return pack_weight_->InitPackWeight(static_cast<const void *>(model_buf), model_size, id, numa_id);
#endif
  return RET_OK;
}

char *PackWeightManager::GetSharedModelBuf(const char *model_buf, std::string model_id,
                                           const std::map<std::string, std::map<std::string, std::string>> *config_info,
                                           bool *is_shared) {
#ifdef SHARING_MODEL_WEIGHT
  std::unique_lock<std::mutex> l(manager_mutex_);
  std::string id = ParseRunnerId(config_info);
  int numa_id = std::atoi(ParseNumaId(config_info).c_str());
  if (id.empty()) {
    MS_LOG(INFO) << "model use share pack weight.";
    id = model_id;
  }
  auto new_model_buf = pack_weight_->GetSharedModelBuf(id, numa_id);
  *is_shared = true;
  return new_model_buf;
#endif
  MS_LOG(INFO) << "model buf not shared.";
  *is_shared = false;
  return const_cast<char *>(model_buf);
}

STATUS PackWeightManager::StoreOriginTensorData(Model *model, std::vector<Tensor *> *all_tensors) {
#ifdef SHARING_MODEL_WEIGHT
  MS_CHECK_TRUE_MSG(model != nullptr, RET_ERROR, "model is nullptr in pack weight manager.");
  if (pack_weight_ == nullptr) {
    MS_LOG(DEBUG) << "define SHARING_MODEL_WEIGHT but not use parallel predict.";
    return RET_OK;
  }
  auto lite_model = reinterpret_cast<LiteModel *>(model);
  auto kernel_num = model->graph_.all_nodes_.size();
  for (size_t i = 0; i < kernel_num; i++) {
    auto node = model->graph_.all_nodes_[i];
    for (size_t j = 0; j < node->input_indices_.size(); j++) {
      auto tensor_index = node->input_indices_[j];
      auto src_tensor = lite_model->GetSchemaTensor(tensor_index);
      if (src_tensor == nullptr || src_tensor->handler() == nullptr || src_tensor->data() == nullptr ||
          src_tensor->length() == 0) {
        continue;
      }
      if (all_tensors->at(tensor_index)->own_data()) {
        auto status = pack_weight_->ReplaceOriginTensorData(lite_model->buf, all_tensors, tensor_index);
        if (status != RET_OK) {
          MS_LOG(DEBUG) << "ReplaceOriginTensorData failed.";
          return RET_ERROR;
        }
      }
      auto status = pack_weight_->StoreOriginTensorData(lite_model->buf, all_tensors->at(tensor_index)->data());
      if (status != RET_OK) {
        MS_LOG(DEBUG) << "data not packed.";
        return RET_ERROR;
      }
    }
  }
#endif
  return RET_OK;
}

void *PackWeightManager::ReplaceFp16Data(void *origin_fp16_data, size_t size, bool *replace) {
#ifdef SHARING_MODEL_WEIGHT
  *replace = true;
  return pack_weight_->ReplaceFp16Data(origin_fp16_data, size);
#endif
  *replace = false;
  return nullptr;
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
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc failed.";
    return nullptr;
  }
#elif defined(__ANDROID__)
  data = malloc(size);
  if (data == nullptr) {
    MS_LOG(ERROR) << "malloc failed.";
    return nullptr;
  }
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

void PackWeightManager::FreePackWeight(std::string id) {
#ifdef SHARING_MODEL_WEIGHT
  std::unique_lock<std::mutex> l(manager_mutex_);
  if (pack_weight_ != nullptr) {
    MS_LOG(INFO) << "free pack weight of " << id;
    pack_weight_->FreePackWeight(id);
    auto it = find(model_ids_.begin(), model_ids_.end(), id);
    if (it != model_ids_.end()) {
      model_ids_.erase(it);
    }
    it = find(runner_ids_.begin(), runner_ids_.end(), id);
    if (it != runner_ids_.end()) {
      runner_ids_.erase(it);
    }
  }
#endif
  return;
}
}  // namespace mindspore::lite
