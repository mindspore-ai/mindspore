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

#include "c_api/include/context.h"
#include "c_api/src/common.h"
#include "c_api/src/resource_manager.h"
#include "utils/ms_context.h"

ResMgrHandle MSResourceManagerCreate() {
  auto res_mgr_ptr = new (std::nothrow) ResourceManager();
  if (res_mgr_ptr == nullptr) {
    MS_LOG(ERROR) << "Failed to allocate Resource Manager!";
    return nullptr;
  }
  return res_mgr_ptr;
}

void MSResourceManagerDestroy(ResMgrHandle res_mgr) {
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  delete res_mgr_ptr;
  res_mgr_ptr = nullptr;
  return;
}

void MSSetEagerMode(bool eager_mode) {
  int mode = eager_mode ? mindspore::kPynativeMode : mindspore::kGraphMode;
  MS_LOG(WARNING) << "Set Execution mode: " << mode;
  auto context = mindspore::MsContext::GetInstance();
  context->set_param<int>(mindspore::MS_CTX_EXECUTION_MODE, mode);
  return;
}

STATUS MSSetBackendPolicy(const char *policy) {
  MS_LOG(WARNING) << "Set Backend Policy: " << policy;
  auto context = mindspore::MsContext::GetInstance();
  return context->set_backend_policy(policy) ? RET_OK : RET_ERROR;
}

void MSSetDeviceTarget(const char *device) {
  MS_LOG(WARNING) << "Set Device Target: " << device;
  auto context = mindspore::MsContext::GetInstance();
  context->set_param<std::string>(mindspore::MS_CTX_DEVICE_TARGET, device);
  return;
}

STATUS MSGetDeviceTarget(char str_buf[], size_t str_len) {
  if (str_buf == nullptr) {
    MS_LOG(ERROR) << "Input char array [str_buf] is nullptr.";
    return RET_NULL_PTR;
  }
  auto context = mindspore::MsContext::GetInstance();
  auto device = context->get_param<std::string>(mindspore::MS_CTX_DEVICE_TARGET);
  size_t valid_size = device.size() < str_len - 1 ? device.size() : str_len - 1;
  for (size_t i = 0; i < valid_size; i++) {
    str_buf[i] = device.c_str()[i];
  }
  str_buf[valid_size] = '\0';
  return RET_OK;
}

void MSSetDeviceId(uint32_t deviceId) {
  MS_LOG(WARNING) << "Set Device ID: " << deviceId;
  auto context = mindspore::MsContext::GetInstance();
  context->set_param<std::uint32_t>(mindspore::MS_CTX_DEVICE_ID, deviceId);
  return;
}

void MSSetGraphsSaveMode(int save_mode) {
  MS_LOG(DEBUG) << "Set Graphs Save Mode: " << save_mode;
  auto context = mindspore::MsContext::GetInstance();
  context->set_param<int>(mindspore::MS_CTX_SAVE_GRAPHS_FLAG, save_mode);
  return;
}

void MSSetGraphsSavePath(const char *save_path) {
  MS_LOG(DEBUG) << "Set Graphs Save Path: " << save_path;
  auto context = mindspore::MsContext::GetInstance();
  context->set_param<std::string>(mindspore::MS_CTX_SAVE_GRAPHS_PATH, save_path);
  return;
}

void MSSetInfer(ResMgrHandle res_mgr, bool infer) {
  MS_LOG(DEBUG) << "Set Infer Graph: " << infer;
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  res_mgr_ptr->SetInfer(infer);
  return;
}

bool MSGetInfer(ResMgrHandle res_mgr) {
  auto res_mgr_ptr = reinterpret_cast<ResourceManager *>(res_mgr);
  return res_mgr_ptr->GetInfer();
}
