/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_ge/ge_context_manager.h"
#include "src/common/log_adapter.h"

namespace mindspore {
GeContextManager::GeContextManager() {}

GeContextManager::~GeContextManager() { DestroyContext(); }

bool GeContextManager::InitContext(uint32_t device_id) {
  device_id_ = device_id;
  auto ret = aclrtSetDevice(device_id_);
  if (ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call aclrtSetDevice , device id " << device_id_ << ", ret: " << static_cast<int>(ret);
    return false;
  }
  // Context will be created by aclrtSetDevice
  ret = aclrtGetCurrentContext(&context_);
  if (ret != ACL_RT_SUCCESS || context_ == nullptr) {
    MS_LOG(ERROR) << "Call aclrtGetCurrentContext failed, ret[" << ret << "]";
    return false;
  }
  MS_LOG(INFO) << "Open device " << device_id_ << " success";
  MS_LOG(INFO) << "Create context success";
  if (!CreateDefaultStream()) {
    MS_LOG(ERROR) << "Failed to create default stream";
    return false;
  }
  return true;
}

bool GeContextManager::SetContext() {
  auto rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_RT_SUCCESS) {
    MS_LOG(ERROR) << "Failed to call aclrtSetCurrentContext";
    return false;
  }
  return true;
}

void GeContextManager::DestroyContext() {
  if (context_) {
    (void)SetContext();
    DestroyDefaultStream();
    context_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy context";
}

aclrtStream GeContextManager::GetDefaultStream() {
  if (default_stream_ != nullptr) {
    return default_stream_;
  }
  if (!CreateDefaultStream()) {
    return nullptr;
  }
  return default_stream_;
}

bool GeContextManager::CreateDefaultStream() {
  if (default_stream_ != nullptr) {
    return true;
  }

  auto priority = 0;
  auto ret = aclrtCreateStreamWithConfig(&default_stream_, priority, (ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC));
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Create stream failed, ret:" << ret;
    return false;
  }
  ret = aclrtSetStreamFailureMode(default_stream_, ACL_STOP_ON_FAILURE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "aclrtSetStreamFailureMode failed, ret:" << ret;
    return false;
  }
  return true;
}

bool GeContextManager::SyncStream(aclrtStream stream) const {
  MS_EXCEPTION_IF_NULL(stream);
  auto RET = aclrtSynchronizeStream(stream);
  if (RET != ACL_ERROR_NONE && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime aclrtSynchronizeStream error.";
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    MS_LOG(WARNING) << "Call runtime aclrtSynchronizeStream, the stream get overflow.";
  }
  return true;
}

void GeContextManager::DestroyDefaultStream() {
  if (default_stream_ == nullptr) {
    return;
  }
  const auto ret = aclrtDestroyStream(default_stream_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtDestroyStream, ret[" << ret << "]";
    return;
  }
  default_stream_ = nullptr;
}
}  // namespace mindspore
