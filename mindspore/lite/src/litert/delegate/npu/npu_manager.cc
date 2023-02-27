/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/npu_manager.h"
#include <sys/system_properties.h>
#include <regex>
#include "include/hiai_ir_build.h"
#include "include/HiAiModelManagerService.h"
#include "src/common/file_utils.h"

namespace mindspore::lite {
constexpr int MAX_MODEL_NUM = 20;
constexpr int KIRIN_REGEX_MIN_SIZE = 2;
constexpr int KIRIN_VERSION_810 = 810;
constexpr int KIRIN_VERSION_820 = 820;
constexpr int KIRIN_VERSION_985 = 985;
int NPUManager::CompareVersion(const std::string &version1, const std::string &version2) {
  std::istringstream iss1(version1);
  std::istringstream iss2(version2);
  std::string string1;
  std::string string2;
  while (!iss1.eof() || !iss2.eof()) {
    getline(iss1, string1, '.');
    getline(iss2, string2, '.');
    if (stoi(string1) > stoi(string2)) return 1;
    if (stoi(string1) < stoi(string2)) return -1;
    string1 = string2 = "0";
  }
  return 0;
}

bool NPUManager::CheckEMUIVersion() {
  char emui[128] = {0x00};
  __system_property_get("ro.build.version.emui", emui);
  std::string emui_str = emui;
  size_t pos = emui_str.find('_');
  if (pos != std::string::npos) {
    auto version = emui_str.substr(pos + 1);
    int ret = CompareVersion(version, "10.0.0");
    if (ret < 0) {
      MS_LOG(WARNING) << "EMUI version " << version << " less than 10.0.0";
      return false;
    }
  }
  return true;
}

void NPUManager::Reset() {
  for (auto client : clients_) {
    client->UnLoadModel();
    client.reset();
  }
  clients_.clear();
  subgraph_index_ = 0;
  domi::HiaiIrBuild ir_build;
  for (const auto &model_map : models_) {
    auto model = model_map.second;
    if (!model->is_freed_) {
      ir_build.ReleaseModelBuff(*model->model_buffer_data_);
      model->is_freed_ = true;
    }
    model->model_buffer_data_.reset();
    model->desc_.reset();
    model->client_.reset();
  }
  models_.clear();
}

bool NPUManager::CheckDDKVerGreatEqual(const std::string &spec_version) {
  auto client = std::make_shared<hiai::AiModelMngerClient>();
  if (client->GetVersion() != nullptr) {
    std::string version = client->GetVersion();
    int ret = CompareVersion(version, spec_version);
    if (ret < 0) {
      MS_LOG(WARNING) << "DDK Version " << version << " less than " << spec_version;
      return false;
    }
  } else {
    MS_LOG(WARNING) << "Get DDK Version failed!";
    return false;
  }
  return true;
}

bool NPUManager::IsSupportNPU() {
  // Avoid multiple checks
  if (!is_check_version_) {
    is_check_version_ = true;
    if (IsKirinChip() && CheckDDKVerGreatEqual("100.320.011.019")) {
      is_support_npu_ = true;
      MS_LOG(INFO) << "The current device support NPU.";
    } else {
      is_support_npu_ = false;
      MS_LOG(WARNING) << "The current device NOT SUPPORT NPU.";
    }
    return is_support_npu_;
  } else {
    return is_support_npu_;
  }
}

bool NPUManager::IsKirinChip() {
  char platform_info[PROP_VALUE_MAX] = {0};
  if (__system_property_get("ro.hardware", platform_info) <= 0) {
    MS_LOG(WARNING) << "Get board platform failed.";
    return false;
  }
  std::string platform_info_str = std::string(platform_info);
  std::cmatch match_result;
  // to match kirin985/kirin990/kirin990 5g/kirin9000/kirin9000E
  std::regex kirin_chip("kirin([0-9]+)[A-Z]*");
  auto ret = std::regex_match(platform_info_str.c_str(), match_result, kirin_chip);
  if (!ret || match_result.size() < KIRIN_REGEX_MIN_SIZE || match_result[1].length() == 0) {
    MS_LOG(WARNING) << "The board platform is not a kirin chip.";
    return false;
  }
  int kirin_number = std::stoi(match_result[1]);
  return kirin_number >= KIRIN_VERSION_985 || kirin_number == KIRIN_VERSION_810 || kirin_number == KIRIN_VERSION_820;
}

int NPUManager::AddModel(std::shared_ptr<domi::ModelBufferData> model_buffer_data, const std::string &model_name,
                         int frequency) {
  auto model = std::make_shared<SubGraphModel>(subgraph_index_, model_name, model_buffer_data);
  auto desc = std::make_shared<hiai::AiModelDescription>(model_name, frequency, 0, 0, 0);
  model->desc_ = desc;
  models_.insert({model_name, model});
  subgraph_index_++;
  return RET_OK;
}

std::shared_ptr<hiai::AiModelMngerClient> NPUManager::CreateAiModelMngerClient() {
  auto client = std::make_shared<hiai::AiModelMngerClient>();
  if (client == nullptr) {
    MS_LOG(ERROR) << "NPU client is nullptr.";
    return nullptr;
  }
  int ret = client->Init(nullptr);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "NPU client init failed. code is " << ret;
    return nullptr;
  }
  return client;
}

int NPUManager::LoadOMModel() {
  std::vector<std::shared_ptr<hiai::AiModelDescription>> models_desc;
  std::shared_ptr<hiai::AiModelMngerClient> client = nullptr;
  std::shared_ptr<hiai::AiModelBuilder> mc_builder = nullptr;
  std::unordered_map<std::shared_ptr<hiai::AiModelBuilder>, hiai::MemBuffer *> builder_buffer_map;
  int total = 0;
  for (const auto &model_map : models_) {
    if (total % MAX_MODEL_NUM == 0) {
      client = CreateAiModelMngerClient();
      if (client == nullptr) {
        MS_LOG(ERROR) << "Create Client failed.";
        return RET_ERROR;
      }
      mc_builder = std::make_shared<hiai::AiModelBuilder>(client);
      if (mc_builder == nullptr) {
        MS_LOG(ERROR) << "Create AiModelBuilder failed.";
        return RET_ERROR;
      }
    }
    total++;
    auto model = model_map.second;
    if (model->is_loaded_ && model->is_freed_) {
      continue;
    }
    models_desc.push_back(model->desc_);

    auto buffer = mc_builder->InputMemBufferCreate(model->model_buffer_data_->data, model->model_buffer_data_->length);
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "NPU input memory buffer create failed.";
      return RET_ERROR;
    }
    builder_buffer_map.insert({mc_builder, buffer});
    model->desc_->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());
    if (models_desc.size() == MAX_MODEL_NUM) {
      auto ret = LoadModel(client, models_desc);
      if (ret != RET_ERROR) {
        MS_LOG(ERROR) << "Client load model failed.";
        return RET_ERROR;
      }
      models_desc.clear();
    }
  }

  if (!models_desc.empty()) {
    auto ret = LoadModel(client, models_desc);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Client load model failed.";
      return RET_ERROR;
    }
    models_desc.clear();
  }

  for (auto it : builder_buffer_map) {
    it.first->MemBufferDestroy(it.second);
  }
  builder_buffer_map.clear();
  return RET_OK;
}

std::shared_ptr<hiai::AiModelMngerClient> NPUManager::GetClient(const std::string &model_name) {
  if (models_.find(model_name) == models_.end() || models_[model_name] == nullptr) {
    return nullptr;
  }
  return models_[model_name]->client_;
}

int NPUManager::LoadModel(const std::shared_ptr<hiai::AiModelMngerClient> &client,
                          std::vector<std::shared_ptr<hiai::AiModelDescription>> desc_list) {
  auto ret = client->Load(desc_list);
  if (ret != hiai::AI_SUCCESS) {
    MS_LOG(ERROR) << "Client load model failed." << ret;
    return RET_ERROR;
  }

  for (const auto &desc : desc_list) {
    auto it = models_.find(desc->GetName());
    it->second->is_loaded_ = true;
    it->second->client_ = client;
  }

  this->clients_.push_back(client);
  return RET_OK;
}
}  // namespace mindspore::lite
