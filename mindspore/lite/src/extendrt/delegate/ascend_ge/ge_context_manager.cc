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
#include "runtime/rt.h"
#include "acl/acl_rt.h"

namespace mindspore {
GeContextManager::GeContextManager() {}

GeContextManager::~GeContextManager() { DestroyContext(); }

bool GeContextManager::InitContext(uint32_t device_id) {
  device_id_ = device_id;
  auto ret = rtSetDevice(device_id_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to call rtSetDevice , device id " << device_id_ << ", ret: " << static_cast<int>(ret);
    return false;
  }
  // Context will be created by rtSetDevice
  ret = rtCtxGetCurrent(&context_);
  if (ret != RT_ERROR_NONE || context_ == nullptr) {
    MS_LOG(ERROR) << "Call rtCtxGetCurrent failed, ret[" << ret << "]";
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
  auto rt_ret = rtCtxSetCurrent(context_);
  if (rt_ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Failed to call rtCtxSetCurrent";
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

rtStream_t GeContextManager::GetDefaultStream() {
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
  auto flags = RT_STREAM_HUGE;
  auto ret = rtStreamCreateWithFlags(&default_stream_, priority, flags);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Create stream failed, ret:" << ret;
    return false;
  }
  ret = rtStreamSetMode(default_stream_, 1);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "rtStreamSetMode failed, ret:" << ret;
    return false;
  }
  return true;
}

bool GeContextManager::SyncStream(rtStream_t stream) const {
  MS_EXCEPTION_IF_NULL(stream);
  auto RET = rtStreamSynchronize(stream);
  if (RET != RT_ERROR_NONE && RET != ACL_ERROR_RT_AICORE_OVER_FLOW) {  // o for switch stream
    MS_LOG(ERROR) << "Call runtime rtStreamSynchronize error.";
    return false;
  }
  if (RET == ACL_ERROR_RT_AICORE_OVER_FLOW) {
    MS_LOG(WARNING) << "Call runtime rtStreamSynchronize, the stream get overflow.";
  }
  return true;
}

void GeContextManager::DestroyDefaultStream() {
  if (default_stream_ == nullptr) {
    return;
  }
  const auto ret = aclrtDestroyStream(default_stream_);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtDestroyStream, ret[" << ret << "]";
    return;
  }
  default_stream_ = nullptr;
}
}  // namespace mindspore
