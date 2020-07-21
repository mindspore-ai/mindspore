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

#include <memory>
#include <algorithm>
#include "serving/acl/acl_session.h"
#include "include/infer_log.h"

namespace mindspore::inference {

std::shared_ptr<InferSession> InferSession::CreateSession(const std::string &device, uint32_t device_id) {
  try {
    auto session = std::make_shared<AclSession>();
    auto ret = session->InitEnv(device, device_id);
    if (!ret) {
      return nullptr;
    }
    return session;
  } catch (std::exception &e) {
    MSI_LOG_ERROR << "Inference CreatSession failed";
    return nullptr;
  }
}

bool AclSession::LoadModelFromFile(const std::string &file_name, uint32_t &model_id) {
  return model_process_.LoadModelFromFile(file_name, model_id);
}

bool AclSession::UnloadModel(uint32_t model_id) {
  model_process_.UnLoad();
  return true;
}

bool AclSession::ExecuteModel(uint32_t model_id, const RequestBase &request,
                              ReplyBase &reply) {  // set d context
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "set the ascend device context failed";
    return false;
  }
  return model_process_.Execute(request, reply);
}

bool AclSession::InitEnv(const std::string &device_type, uint32_t device_id) {
  device_type_ = device_type;
  device_id_ = device_id;
  auto ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Execute aclInit Failed";
    return false;
  }
  MSI_LOG_INFO << "acl init success";

  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl open device " << device_id_ << " failed";
    return false;
  }
  MSI_LOG_INFO << "open device " << device_id_ << " success";

  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl create context failed";
    return false;
  }
  MSI_LOG_INFO << "create context success";

  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl create stream failed";
    return false;
  }
  MSI_LOG_INFO << "create stream success";

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl get run mode failed";
    return false;
  }
  bool is_device = (run_mode == ACL_DEVICE);
  model_process_.SetIsDevice(is_device);
  MSI_LOG_INFO << "get run mode success is device input/output " << is_device;

  MSI_LOG_INFO << "Init acl success, device id " << device_id_;
  return true;
}

bool AclSession::FinalizeEnv() {
  aclError ret;
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "destroy stream failed";
    }
    stream_ = nullptr;
  }
  MSI_LOG_INFO << "end to destroy stream";
  if (context_ != nullptr) {
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
      MSI_LOG_ERROR << "destroy context failed";
    }
    context_ = nullptr;
  }
  MSI_LOG_INFO << "end to destroy context";

  ret = aclrtResetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "reset devie " << device_id_ << " failed";
  }
  MSI_LOG_INFO << "end to reset device " << device_id_;

  ret = aclFinalize();
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "finalize acl failed";
  }
  MSI_LOG_INFO << "end to finalize acl";
  return true;
}

AclSession::AclSession() = default;
}  // namespace mindspore::inference
