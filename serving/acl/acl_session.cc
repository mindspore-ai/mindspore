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
#include <fstream>
#include "serving/acl/acl_session.h"
#include "include/infer_log.h"

namespace mindspore::inference {

std::shared_ptr<InferSession> InferSession::CreateSession(const std::string &device, uint32_t device_id) {
  try {
    auto session = std::make_shared<AclSession>();
    auto ret = session->InitEnv(device, device_id);
    if (ret != SUCCESS) {
      return nullptr;
    }
    return session;
  } catch (std::exception &e) {
    MSI_LOG_ERROR << "Inference CreatSession failed";
    return nullptr;
  }
}

Status AclSession::LoadModelFromFile(const std::string &file_name, uint32_t &model_id) {
  Status ret = model_process_.LoadModelFromFile(file_name, model_id);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "Load model from file failed, model file " << file_name;
    return FAILED;
  }
  std::string dvpp_config_file;
  auto index = file_name.rfind(".");
  if (index == std::string::npos) {
    dvpp_config_file = file_name;
  } else {
    dvpp_config_file = file_name.substr(0, index);
  }
  dvpp_config_file += "_dvpp_config.json";
  std::ifstream fp(dvpp_config_file);
  if (!fp.is_open()) {
    MSI_LOG_INFO << "Dvpp config file not exist, model will execute with tensors as inputs, dvpp config file "
                 << dvpp_config_file;
    return SUCCESS;
  }
  fp.close();
  if (dvpp_process_.InitWithJsonConfig(dvpp_config_file) != SUCCESS) {
    MSI_LOG_ERROR << "Dvpp config file parse error, dvpp config file " << dvpp_config_file;
    return FAILED;
  }
  execute_with_dvpp_ = true;
  MSI_LOG_INFO << "Dvpp config success";
  return SUCCESS;
}

Status AclSession::UnloadModel(uint32_t /*model_id*/) {
  model_process_.UnLoad();
  return SUCCESS;
}

Status AclSession::ExecuteModel(uint32_t /*model_id*/, const RequestBase &request,
                                ReplyBase &reply) {  // set d context
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "set the ascend device context failed";
    return FAILED;
  }
  return model_process_.Execute(request, reply);
}

Status AclSession::PreProcess(uint32_t /*model_id*/, const InferImagesBase *images_input,
                              ImagesDvppOutput &dvpp_output) {
  if (images_input == nullptr) {
    MSI_LOG_ERROR << "images input is nullptr";
    return FAILED;
  }
  auto batch_size = images_input->batch_size();
  if (batch_size <= 0) {
    MSI_LOG_ERROR << "invalid batch size " << images_input->batch_size();
    return FAILED;
  }
  std::vector<const void *> pic_buffer_list;
  std::vector<size_t> pic_size_list;
  for (size_t i = 0; i < batch_size; i++) {
    const void *pic_buffer = nullptr;
    uint32_t pic_size = 0;
    if (!images_input->get(i, pic_buffer, pic_size) || pic_buffer == nullptr || pic_size == 0) {
      MSI_LOG_ERROR << "Get request " << 0 << "th buffer failed";
      return FAILED;
    }
    pic_buffer_list.push_back(pic_buffer);
    pic_size_list.push_back(pic_size);
  }
  auto ret = dvpp_process_.Process(pic_buffer_list, pic_size_list, dvpp_output.buffer_device, dvpp_output.buffer_size);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "dvpp process failed";
    return ret;
  }
  return SUCCESS;
}

Status AclSession::ExecuteModel(uint32_t model_id, const ImagesRequestBase &images_inputs,  // images for preprocess
                                const RequestBase &request, ReplyBase &reply) {
  if (!execute_with_dvpp_) {
    MSI_LOG_ERROR << "Unexpected images as inputs, DVPP not config";
    return INFER_STATUS(INVALID_INPUTS) << "Unexpected images as inputs, DVPP not config";
  }
  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "set the ascend device context failed";
    return FAILED;
  }
  if (images_inputs.size() != 1) {
    MSI_LOG_ERROR << "Only support one input to do DVPP preprocess";
    return INFER_STATUS(INVALID_INPUTS) << "Only support one input to do DVPP preprocess";
  }
  if (images_inputs[0] == nullptr) {
    MSI_LOG_ERROR << "Get first images input failed";
    return FAILED;
  }
  if (images_inputs[0]->batch_size() != model_process_.GetBatchSize()) {
    MSI_LOG_ERROR << "Input batch size " << images_inputs[0]->batch_size() << " not match Model batch size "
                  << model_process_.GetBatchSize();
    return INFER_STATUS(INVALID_INPUTS) << "Input batch size " << images_inputs[0]->batch_size()
                                        << " not match Model batch size " << model_process_.GetBatchSize();
  }
  if (request.size() != 0) {
    MSI_LOG_ERROR << "only support one input, images input size is 1, tensor inputs is not 0 " << request.size();
    return INFER_STATUS(INVALID_INPUTS) << "only support one input, images input size is 1, tensor inputs is not 0 "
                                        << request.size();
  }
  ImagesDvppOutput dvpp_output;
  Status ret = PreProcess(model_id, images_inputs[0], dvpp_output);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "DVPP preprocess failed";
    return ret;
  }
  ret = model_process_.Execute(dvpp_output.buffer_device, dvpp_output.buffer_size, reply);
  if (ret != SUCCESS) {
    MSI_LOG_ERROR << "Execute model failed";
    return ret;
  }
  return SUCCESS;
}

Status AclSession::InitEnv(const std::string &device_type, uint32_t device_id) {
  device_type_ = device_type;
  device_id_ = device_id;
  auto ret = aclInit(nullptr);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "Execute aclInit Failed";
    return FAILED;
  }
  MSI_LOG_INFO << "acl init success";

  ret = aclrtSetDevice(device_id_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl open device " << device_id_ << " failed";
    return FAILED;
  }
  MSI_LOG_INFO << "open device " << device_id_ << " success";

  ret = aclrtCreateContext(&context_, device_id_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl create context failed";
    return FAILED;
  }
  MSI_LOG_INFO << "create context success";

  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl create stream failed";
    return FAILED;
  }
  MSI_LOG_INFO << "create stream success";

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MSI_LOG_ERROR << "acl get run mode failed";
    return FAILED;
  }
  bool is_device = (run_mode == ACL_DEVICE);
  model_process_.SetIsDevice(is_device);
  MSI_LOG_INFO << "get run mode success is device input/output " << is_device;

  if (dvpp_process_.InitResource(stream_) != SUCCESS) {
    MSI_LOG_ERROR << "dvpp init resource failed";
    return FAILED;
  }
  MSI_LOG_INFO << "Init acl success, device id " << device_id_;
  return SUCCESS;
}

Status AclSession::FinalizeEnv() {
  dvpp_process_.Finalize();
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
  return SUCCESS;
}

AclSession::AclSession() = default;
}  // namespace mindspore::inference
