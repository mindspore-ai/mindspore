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
    int ret = CompareVersion(version, "11.0.0");
    if (ret < 0) {
      return false;
    }
  }
  return true;
}

bool NPUManager::CheckDDKVersion() {
  auto client = std::make_shared<hiai::AiModelMngerClient>();
  if (client->GetVersion() != nullptr) {
    std::string version = client->GetVersion();
    int ret = CompareVersion(version, "100.330.010.011");
    if (ret < 0) {
      return false;
    }
  }
  return true;
}
bool NPUManager::IsSupportNPU() {
  if (IsKirinChip() && CheckEMUIVersion() && CheckDDKVersion()) {
    MS_LOG(INFO) << "The current device support NPU.";
    return true;
  } else {
    MS_LOG(INFO) << "The current device NOT SUPPORT NPU.";
    return false;
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
    auto kirin_number_str = line.substr(index + 5);
    auto kirin_number = atoi(kirin_number_str.c_str());
    if (kirin_number >= 985 || kirin_number == 810 || kirin_number == 820) {
      cpu_info.close();
      return true;
    } else {
      cpu_info.close();
      return false;
    }
  }
  return false;
}

int NPUManager::AddModel(void *model_buf, uint32_t size, const std::string &model_name, int frequency) {
  hiai::MemBuffer *buffer = mc_builder_->InputMemBufferCreate(model_buf, size);
  if (buffer == nullptr) {
    MS_LOG(ERROR) << "MemBuffer is null.";
    return RET_ERROR;
  }

  auto desc = std::make_shared<hiai::AiModelDescription>(model_name, frequency, 0, 0, 0);
  desc->SetModelBuffer(buffer->GetMemBufferData(), buffer->GetMemBufferSize());
  model_desc_.push_back(desc);
  mc_builder_->MemBufferDestroy(buffer);

  model_map_.insert({model_name, index_});
  index_++;
  return RET_OK;
}

int NPUManager::LoadOMModel() {
  for (int i = 0; i < index_ / MAX_MODEL_NUM + 1; i++) {
    auto client = std::make_shared<hiai::AiModelMngerClient>();
    if (client == nullptr) {
      MS_LOG(ERROR) << "NPU client is nullptr.";
      return RET_ERROR;
    }
    int ret = client->Init(nullptr);
    if (ret != hiai::AI_SUCCESS) {
      MS_LOG(ERROR) << "NPU client init failed. code is " << ret;
      return RET_ERROR;
    }
    mc_builder_ = std::make_shared<hiai::AiModelBuilder>(client);

    vector<std::shared_ptr<hiai::AiModelDescription>> desc(model_desc_.begin() + i * MAX_MODEL_NUM,
                                                           ((i + 1) * MAX_MODEL_NUM > index_)
                                                             ? model_desc_.begin() + index_
                                                             : model_desc_.begin() + (i + 1) * MAX_MODEL_NUM);
    ret = client->Load(desc);
    if (ret != hiai::AI_SUCCESS) {
      MS_LOG(ERROR) << "Client load model failed." << ret;
      return RET_ERROR;
    }
    clients_.push_back(client);
  }
  return RET_OK;
}

std::shared_ptr<hiai::AiModelMngerClient> NPUManager::GetClient(const std::string &model_name) {
  return clients_[model_map_[model_name] / MAX_MODEL_NUM];
}

int NPUManager::index() const { return index_; }
}  // namespace mindspore::lite
