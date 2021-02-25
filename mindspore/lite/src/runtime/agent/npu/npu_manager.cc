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

#include "src/runtime/agent/npu/npu_manager.h"
#include <sys/system_properties.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include "include/hiai_ir_build.h"
#include "include/HiAiModelManagerService.h"
#include "include/errorcode.h"
#include "src/common/file_utils.h"

namespace mindspore::lite {
#define MAX_MODEL_NUM 20
int NPUManager::CompareVersion(const string &version1, const string &version2) {
  std::istringstream iss1(version1);
  std::istringstream iss2(version2);
  string string1;
  string string2;
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
  int pos = emui_str.find('_');
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

  index_ = 0;
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

bool NPUManager::CheckDDKVersion() {
  auto client = std::make_shared<hiai::AiModelMngerClient>();
  if (client->GetVersion() != nullptr) {
    std::string version = client->GetVersion();
    int ret = CompareVersion(version, "100.320.010.023");
    if (ret < 0) {
      MS_LOG(WARNING) << "DDK Version " << version << " less than 100.320.010.023";
      return false;
    }
  }
  return true;
}
bool NPUManager::IsSupportNPU() {
  // Avoid multiple checks
  if (!is_check_version_) {
    is_check_version_ = true;
    if (IsKirinChip() && CheckEMUIVersion() && CheckDDKVersion()) {
      is_support_ = true;
      MS_LOG(INFO) << "The current device support NPU.";
    } else {
      is_support_ = false;
      MS_LOG(WARNING) << "The current device NOT SUPPORT NPU.";
    }
    return is_support_;
  } else {
    return is_support_;
  }
}

bool NPUManager::IsKirinChip() {
  std::ifstream cpu_info("/proc/cpuinfo");
  if (!(cpu_info.good() && cpu_info.is_open())) {
    return false;
  }
  std::string line;
  while (!cpu_info.eof()) {
    getline(cpu_info, line);
    if (line.find("Hardware") == string::npos) {
      continue;
    }
    auto index = line.find("Kirin");
    if (index == string::npos) {
      continue;
    }
    // support Kirin 990 5G\990E\9000E
    if (line.find("990") != string::npos || line.find("9000") != string::npos) {
      cpu_info.close();
      return true;
    }
    auto kirin_number_str = line.substr(index + 5);
    auto kirin_number = atoi(kirin_number_str.c_str());
    if (kirin_number >= 985 || kirin_number == 810 || kirin_number == 820) {
      cpu_info.close();
      return true;
    } else {
      MS_LOG(WARNING) << "Unsupported KirinChip " << kirin_number;
      cpu_info.close();
      return false;
    }
  }
  return false;
}

int NPUManager::AddModel(std::shared_ptr<domi::ModelBufferData> model_buffer_data, const std::string &model_name,
                         int frequency) {
  auto model = std::make_shared<SubGraphModel>(index_, model_name, model_buffer_data);
  auto desc = std::make_shared<hiai::AiModelDescription>(model_name, frequency, 0, 0, 0);
  model->desc_ = desc;
  models_.insert({model_name, model});
  index_++;
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

int NPUManager::index() const { return index_; }

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
