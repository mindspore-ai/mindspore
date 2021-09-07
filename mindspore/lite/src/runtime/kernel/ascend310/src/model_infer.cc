/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/ascend310/src/model_infer.h"
#include "common/log_adapter.h"
#include "acl/acl.h"

namespace mindspore::kernel {
namespace acl {
ModelInfer::ModelInfer(const Buffer &om_data, const AclModelOptions &options)
    : init_flag_(false),
      load_flag_(false),
      device_type_("AscendCL"),
      context_(nullptr),
      om_data_(om_data),
      options_(options),
      model_process_(),
      acl_env_(nullptr) {}

STATUS ModelInfer::Init() {
  if (init_flag_) {
    MS_LOG(INFO) << "Acl has been initialized, skip.";
    return lite::RET_OK;
  }

  acl_env_ = AclEnvGuard::GetAclEnv(options_.dump_cfg_path);
  if (acl_env_ == nullptr) {
    MS_LOG(ERROR) << "Acl init failed.";
    return lite::RET_ERROR;
  }
  int32_t device_id = options_.device_id;
  aclError ret = aclrtSetDevice(device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl open device " << device_id << " failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Open device " << device_id << " success.";

  ret = aclrtCreateContext(&context_, device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl create context failed.";
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Create context success.";

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl get run mode failed.";
    return lite::RET_ERROR;
  }
  bool is_device = (run_mode == ACL_DEVICE);
  model_process_.SetIsDevice(is_device);
  MS_LOG(INFO) << "Get run mode success is device input/output " << is_device;

  MS_LOG(INFO) << "Init acl success, device id " << device_id;
  init_flag_ = true;
  return lite::RET_OK;
}

STATUS ModelInfer::Finalize() {
  if (!init_flag_) {
    MS_LOG(WARNING) << "Init is not ok, no need to finalize.";
    return lite::RET_OK;
  }

  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed.";
    return lite::RET_ERROR;
  }

  int ret = model_process_.UnLoad();
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "Unload model inner failed.";
    return ret;
  }

  if (context_ != nullptr) {
    rt_ret = aclrtDestroyContext(context_);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy context failed.";
    }
    context_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy context.";

  rt_ret = aclrtResetDevice(options_.device_id);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reset device " << options_.device_id << " failed.";
  }
  MS_LOG(INFO) << "End to reset device " << options_.device_id;
  init_flag_ = false;
  load_flag_ = false;
  return lite::RET_OK;
}

STATUS ModelInfer::Load() {
  if (!load_flag_) {
    int ret = LoadAclModel(om_data_);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Load acl model failed.";
      return ret;
    }
    load_flag_ = true;
  }

  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed, ret = " << rt_ret;
    return lite::RET_ERROR;
  }

  return lite::RET_OK;
}

STATUS ModelInfer::LoadAclModel(const Buffer &om_data) {
  MS_LOG(INFO) << "Start load acl model.";
  // acl load model
  uint32_t acl_model_id;
  auto acl_ret = aclmdlLoadFromMem(om_data.Data(), om_data.DataSize(), &acl_model_id);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
    return lite::RET_ERROR;
  }

  // acl init model resource
  model_process_.set_model_id(acl_model_id);
  int ret = model_process_.PreInitModelResource();
  if (ret != lite::RET_OK) {
    (void)aclmdlUnload(acl_model_id);
    MS_LOG(ERROR) << "Pre init model resource failed.";
    return ret;
  }

  MS_LOG(INFO) << "Load acl model success.";
  return lite::RET_OK;
}

STATUS ModelInfer::Inference(const std::vector<mindspore::MSTensor> &inputs,
                             std::vector<mindspore::MSTensor> *outputs) {
  if (Load() != lite::RET_OK) {
    MS_LOG(ERROR) << "Prepare model resource failed.";
    return lite::RET_ERROR;
  }

  return model_process_.PredictFromHost(inputs, outputs);
}
}  // namespace acl
}  // namespace mindspore::kernel
