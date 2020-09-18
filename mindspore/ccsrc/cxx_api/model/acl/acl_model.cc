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

#include "cxx_api/model/acl/acl_model.h"
#include <memory>
#include "utils/context/context_extends.h"

namespace mindspore::api {
std::weak_ptr<AclModel::AclEnvGuard> AclModel::global_acl_env_;
std::mutex AclModel::global_acl_env_mutex_;

Status AclModel::InitEnv() {
  if (init_flag_) {
    return SUCCESS;
  }

  MS_EXCEPTION_IF_NULL(options_);
  aclError ret;
  {
    std::lock_guard<std::mutex> lock(global_acl_env_mutex_);
    acl_env_ = global_acl_env_.lock();
    if (acl_env_ != nullptr) {
      if (options_->dump_cfg_path.empty()) {
        MS_LOG(INFO) << "Acl has been initialized, skip.";
      } else {
        MS_LOG(WARNING) << "Acl has been initialized, skip, so dump config will be ignored.";
      }
    } else {
      acl_env_ = std::make_shared<AclEnvGuard>(options_->dump_cfg_path);
      if (acl_env_->GetErrno() != ACL_ERROR_NONE) {
        MS_LOG(ERROR) << "Execute aclInit Failed";
        return FAILED;
      }
      global_acl_env_ = acl_env_;
      MS_LOG(INFO) << "Acl init success";
    }
  }

  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl open device " << device_id_ << " failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Open device " << device_id_ << " success";

  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl create context failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Create context success";

  ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl set current context failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Set context success";

  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl create stream failed";
    return FAILED;
  }
  MS_LOG(INFO) << "Create stream success";

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl get run mode failed";
    return FAILED;
  }
  bool is_device = (run_mode == ACL_DEVICE);
  model_process_.SetIsDevice(is_device);
  MS_LOG(INFO) << "Get run mode success is device input/output " << is_device;

  if (dvpp_process_.InitResource(stream_) != SUCCESS) {
    MS_LOG(ERROR) << "DVPP init resource failed";
    return FAILED;
  }
  ModelConverter::RegAllOp();

  MS_LOG(INFO) << "Init acl success, device id " << device_id_;
  init_flag_ = true;
  return SUCCESS;
}

Status AclModel::FinalizeEnv() {
  if (!init_flag_) {
    return SUCCESS;
  }

  dvpp_process_.Finalize();
  aclError ret;
  if (stream_ != nullptr) {
    ret = aclrtDestroyStream(stream_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy stream failed";
    }
    stream_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy stream";
  if (context_ != nullptr) {
    ret = aclrtDestroyContext(context_);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy context failed";
    }
    context_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy context";

  ret = aclrtResetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reset devie " << device_id_ << " failed";
  }
  MS_LOG(INFO) << "End to reset device " << device_id_;

  init_flag_ = false;
  return SUCCESS;
}

Status AclModel::LoadModel(const Buffer &model_data, ModelType type,
                           const std::map<std::string, std::string> &options) {
  if (load_flag_) {
    MS_LOG(ERROR) << "Model has been loaded.";
    return FAILED;
  }

  options_ = std::make_unique<AclModelOptions>(options);
  MS_EXCEPTION_IF_NULL(options_);

  Status ret = InitEnv();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "InitEnv failed.";
    return FAILED;
  }

  Buffer om_data;
  if (type == ModelType::kMindIR) {
    model_converter_.set_options(options_.get());
    om_data = model_converter_.LoadMindIR(model_data);
  } else if (type == ModelType::kAIR) {
    model_converter_.set_options(options_.get());
    om_data = model_converter_.LoadAscendIR(model_data);
  } else if (type == ModelType::kOM) {
    om_data = model_data;
  } else {
    MS_LOG(ERROR) << "Unsupported model type " << type;
    return FAILED;
  }

  // acl load model
  uint32_t acl_model_id;
  auto acl_ret = aclmdlLoadFromMem(om_data.Data(), om_data.DataSize(), &acl_model_id);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed.";
    return FAILED;
  }

  // acl init model resource
  model_process_.set_model_id(acl_model_id);
  ret = model_process_.PreInitModelResource();
  if (ret != SUCCESS) {
    (void)aclmdlUnload(acl_model_id);
    MS_LOG(ERROR) << "Pre init model resource failed.";
    return FAILED;
  }

  // acl init dvpp
  ret = dvpp_process_.InitWithJsonConfig(options_->dvpp_cfg_path);
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "DVPP config file parse error.";
    return FAILED;
  }

  load_flag_ = true;
  return SUCCESS;
}

Status AclModel::LoadModel(const std::string &file_name, ModelType type,
                           const std::map<std::string, std::string> &options) {
  Buffer model_data = ModelConverter::ReadFile(file_name);
  if (model_data.DataSize() == 0) {
    MS_LOG(ERROR) << "Read file " << file_name << " failed.";
    return FAILED;
  }

  return LoadModel(model_data, type, options);
}

Status AclModel::UnloadModel() {
  if (!load_flag_) {
    MS_LOG(WARNING) << "No model is loaded, skip unload.";
    return SUCCESS;
  }

  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed";
    return FAILED;
  }

  Status ret = model_process_.UnLoad();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Unload model inner failed.";
    return FAILED;
  }

  ret = FinalizeEnv();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "FinalizeEnv failed.";
    return FAILED;
  }

  MS_LOG(INFO) << "Unload model success.";
  load_flag_ = false;
  return SUCCESS;
}

Status AclModel::Train(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status AclModel::Eval(const DataSet &, std::map<std::string, Buffer> *) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return FAILED;
}

Status AclModel::Predict(const std::map<std::string, Buffer> &inputs, std::map<std::string, Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  if (!load_flag_) {
    MS_LOG(ERROR) << "No model is loaded, predict failed.";
    return FAILED;
  }

  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed";
    return FAILED;
  }
  return model_process_.Predict(inputs, outputs);
}

Status AclModel::GetInputsInfo(std::vector<Tensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(tensor_list);
  return model_process_.GetInputsInfo(tensor_list);
}

Status AclModel::GetOutputsInfo(std::vector<Tensor> *tensor_list) const {
  MS_EXCEPTION_IF_NULL(tensor_list);
  return model_process_.GetOutputsInfo(tensor_list);
}

AclModel::AclEnvGuard::AclEnvGuard(const std::string &cfg_file) {
  errno_ = aclInit(common::SafeCStr(cfg_file));
  if (errno_ != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Execute aclInit Failed";
    return;
  }
  MS_LOG(INFO) << "Acl init success";
}

AclModel::AclEnvGuard::~AclEnvGuard() {
  errno_ = aclFinalize();
  if (errno_ != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Finalize acl failed";
  }
  MS_LOG(INFO) << "Acl finalize success";
}
}  // namespace mindspore::api
