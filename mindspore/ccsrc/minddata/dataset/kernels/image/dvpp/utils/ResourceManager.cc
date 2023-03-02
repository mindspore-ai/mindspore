/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ResourceManager.h"
#include <algorithm>
#include <memory>
#include <string>

bool ResourceManager::initFlag_ = true;
std::shared_ptr<ResourceManager> ResourceManager::ptr_ = nullptr;

/**
 * Check whether the file exists.
 *
 * @param filePath the file path we want to check
 * @return APP_ERR_OK if file exists, error code otherwise
 */
APP_ERROR ExistFile(const std::string &filePath) {
  struct stat fileSat = {0};
  char c[PATH_MAX] = {0x00};
  size_t count = filePath.copy(c, PATH_MAX);
  if (count != filePath.length()) {
    MS_LOG(ERROR) << "Failed to strcpy" << c;
    return APP_ERR_COMM_FAILURE;
  }
  // Get the absolute path of input directory
  char path[PATH_MAX] = {0x00};
  if ((strlen(c) >= PATH_MAX) || (realpath(c, path) == nullptr)) {
    MS_LOG(ERROR) << "Failed to get canonicalize path";
    return APP_ERR_COMM_EXIST;
  }
  if (stat(c, &fileSat) == 0 && S_ISREG(fileSat.st_mode)) {
    return APP_ERR_OK;
  }
  return APP_ERR_COMM_FAILURE;
}

void ResourceManager::Release() {
  APP_ERROR ret;
  for (size_t i = 0; i < deviceIds_.size(); i++) {
    if (contexts_[i] != nullptr) {
      ret = aclrtDestroyContext(contexts_[i]);  // Destroy context
      if (ret != APP_ERR_OK) {
        MS_LOG(ERROR) << "Failed to destroy context, ret = " << ret << ".";
        return;
      }
      contexts_[i] = nullptr;
    }
    ret = aclrtResetDevice(deviceIds_[i]);  // Reset device
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to reset device, ret = " << ret << ".";
      return;
    }
  }

  // finalize the acl when the process exit
  ret = AclInitAdapter::GetInstance().AclFinalize();
  if (ret != APP_ERR_OK) {
    MS_LOG(DEBUG) << "Failed to finalize acl, ret = " << ret << ".";
  }

  // release all the members
  acl_env_ = nullptr;
  deviceIds_.clear();
  deviceIdMap_.clear();
  ptr_ = nullptr;
  initFlag_ = true;
  contexts_.clear();

  MS_LOG(INFO) << "Release the resource(s) successfully.";
}

std::shared_ptr<ResourceManager> ResourceManager::GetInstance() {
  if (ptr_ == nullptr) {
    ResourceManager *temp = new ResourceManager();
    ptr_.reset(temp);
  }
  return ptr_;
}

APP_ERROR ResourceManager::InitResource(ResourceInfo &resourceInfo) {
  if (acl_env_ != nullptr) {
    MS_LOG(INFO) << "Acl has been initialized, skip.";
    return APP_ERR_OK;
  }
  APP_ERROR ret = APP_ERR_OK;
  acl_env_ = AclEnvGuard::GetAclEnv();
  if (acl_env_ == nullptr) {
    MS_LOG(ERROR) << "Failed to init acl.";
    return APP_ERR_COMM_FAILURE;
  }
  (void)std::copy(resourceInfo.deviceIds.begin(), resourceInfo.deviceIds.end(), std::back_inserter(deviceIds_));
  MS_LOG(INFO) << "Initialized acl successfully.";
  // Open device and create context for each chip, note: it create one context for each chip
  for (size_t i = 0; i < deviceIds_.size(); i++) {
    deviceIdMap_[deviceIds_[i]] = i;
    ret = aclrtSetDevice(deviceIds_[i]);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to open acl device: " << deviceIds_[i];
      return ret;
    }
    MS_LOG(INFO) << "Open device " << deviceIds_[i] << " successfully.";
    aclrtContext context;
    ret = aclrtCreateContext(&context, deviceIds_[i]);
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to create acl context, ret = " << ret << ".";
      return ret;
    }
    MS_LOG(INFO) << "Created context for device " << deviceIds_[i] << " successfully";
    contexts_.push_back(context);
  }
  std::string singleOpPath = resourceInfo.singleOpFolderPath;
  if (!singleOpPath.empty()) {
    ret = aclopSetModelDir(singleOpPath.c_str());  // Set operator model directory for application
    if (ret != APP_ERR_OK) {
      MS_LOG(ERROR) << "Failed to aclopSetModelDir, ret = " << ret << ".";
      return ret;
    }
  }
  MS_LOG(INFO) << "Init resource successfully.";
  ResourceManager::initFlag_ = false;
  return APP_ERR_OK;
}

aclrtContext ResourceManager::GetContext(int deviceId) { return contexts_[deviceIdMap_[deviceId]]; }
