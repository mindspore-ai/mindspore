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

#include "src/litert/kernel/ascend/src/model_infer.h"
#include "common/log_adapter.h"
#include "acl/acl.h"
#include "src/litert/kernel/ascend/src/acl_mem_manager.h"

namespace mindspore::kernel {
namespace acl {
namespace {
constexpr auto kModelSharingPrepareKey = "multi_model_sharing_mem_prepare";
constexpr auto kModelSharingKey = "multi_model_sharing_mem";
}  // namespace

ModelInfer::ModelInfer(const Buffer &om_data, const AclModelOptions &options,
                       const std::map<std::string, std::string> &config_info)
    : init_flag_(false),
      load_flag_(false),
      device_type_("AscendCL"),
      context_(nullptr),
      om_data_(om_data),
      options_(options),
      model_process_(options),
      config_info_(config_info),
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
    MS_LOG(ERROR) << "Acl open device " << device_id << " failed, ret " << ret;
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Open device " << device_id << " success.";

  ret = aclrtCreateContext(&context_, device_id);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl create context failed, ret " << ret;
    return lite::RET_ERROR;
  }
  MS_LOG(INFO) << "Create context success.";

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl get run mode failed, ret " << ret;
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
    MS_LOG(ERROR) << "Set the ascend device context failed, ret " << rt_ret;
    return lite::RET_ERROR;
  }
  if (load_flag_) {
    auto ret = model_process_.UnLoad();
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Unload model inner failed.";
      return ret;
    }
  }
  if (context_ != nullptr) {
    rt_ret = aclrtDestroyContext(context_);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy context failed, ret " << rt_ret;
    }
    context_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy context.";

  rt_ret = aclrtResetDevice(options_.device_id);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reset device " << options_.device_id << " failed, ret " << rt_ret;
  }
  MS_LOG(INFO) << "End to reset device " << options_.device_id;
  init_flag_ = false;
  load_flag_ = false;
  return lite::RET_OK;
}

bool ModelInfer::IsEnableMultiModelSharingMemPrepare() {
  if (config_info_.find(kModelSharingPrepareKey) != config_info_.end()) {
    return true;
  }
  return false;
}

bool ModelInfer::IsEnableMultiModelSharingMem() {
  if (config_info_.find(kModelSharingKey) != config_info_.end()) {
    return true;
  }
  return false;
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
  size_t work_size = 0;
  size_t weight_size = 0;

  if (IsEnableMultiModelSharingMemPrepare()) {
    auto acl_ret = aclmdlQuerySizeFromMem(om_data.Data(), om_data.DataSize(), &work_size, &weight_size);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclmdlQuerySizeFromMem failed, ret = " << acl_ret;
      return lite::RET_ERROR;
    }
    AclMemManager::GetInstance().UpdateWorkspace(work_size, weight_size);
    return lite::RET_OK;
  } else if (IsEnableMultiModelSharingMem()) {
    AclModelMemInfo acl_work_mem_info;
    AclModelMemInfo acl_weight_mem_info;
    auto ret = AclMemManager::GetInstance().GetModelWorkMem(&acl_work_mem_info);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Get work mem failed.";
      return ret;
    }
    ret = AclMemManager::GetInstance().GetModelWeightMem(&acl_weight_mem_info);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Get weight mem failed.";
      return ret;
    }
    MS_LOG(DEBUG) << "Sharing work size = " << acl_work_mem_info.mem_size;
    MS_LOG(DEBUG) << "Sharing weight size = " << acl_weight_mem_info.mem_size;
    auto acl_ret =
      aclmdlLoadFromMemWithMem(om_data.Data(), om_data.DataSize(), &acl_model_id, acl_work_mem_info.mem_addr,
                               acl_work_mem_info.mem_size, acl_weight_mem_info.mem_addr, acl_weight_mem_info.mem_size);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclmdlLoadFromMemWithMem failed, ret = " << acl_ret;
      return lite::RET_ERROR;
    }
    model_process_.SetSharingWorkspaceFlag(true);
  } else {
    auto acl_ret = aclmdlLoadFromMem(om_data.Data(), om_data.DataSize(), &acl_model_id);
    if (acl_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
      return lite::RET_ERROR;
    }
  }

  // acl init model resource
  model_process_.set_model_id(acl_model_id);
  auto ret = model_process_.PreInitModelResource();
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

std::set<uint64_t> ModelInfer::GetDynamicBatch() { return model_process_.GetDynamicBatch(); }

// need to be called after model load;
std::set<std::pair<uint64_t, uint64_t>> ModelInfer::GetDynamicImage() { return model_process_.GetDynamicImage(); }
}  // namespace acl
}  // namespace mindspore::kernel
