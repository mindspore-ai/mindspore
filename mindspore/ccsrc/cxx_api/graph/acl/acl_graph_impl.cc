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
#include "cxx_api/graph/acl/acl_graph_impl.h"
#include "include/api/context.h"
#include "cxx_api/model/acl/model_converter.h"
#include "cxx_api/python_utils.h"
#include "utils/log_adapter.h"

namespace mindspore::api {
API_FACTORY_REG(GraphCell::GraphImpl, Ascend310, AclGraphImpl);

AclGraphImpl::AclGraphImpl()
    : init_flag_(false),
      load_flag_(false),
      device_type_("AscendCL"),
      device_id_(Context::Instance().GetDeviceID()),
      context_(nullptr),
      acl_env_(nullptr) {}

AclGraphImpl::~AclGraphImpl() { (void)FinalizeEnv(); }

Status AclGraphImpl::Run(const std::vector<Buffer> &inputs, std::vector<Buffer> *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  Status ret = Load();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Prepare model resource failed.";
    return FAILED;
  }

  return model_process_.PredictFromHost(inputs, outputs);
}

Status AclGraphImpl::GetInputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                   std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  Status ret = Load();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Prepare model resource failed.";
    return FAILED;
  }

  return model_process_.GetInputsInfo(names, shapes, data_types, mem_sizes);
}

Status AclGraphImpl::GetOutputsInfo(std::vector<std::string> *names, std::vector<std::vector<int64_t>> *shapes,
                                    std::vector<DataType> *data_types, std::vector<size_t> *mem_sizes) {
  Status ret = Load();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "Prepare model resource failed.";
    return FAILED;
  }

  return model_process_.GetOutputsInfo(names, shapes, data_types, mem_sizes);
}

Status AclGraphImpl::LoadAclModel(Buffer om_data) {
  MS_LOG(INFO) << "Start load acl model.";
  // acl load model
  uint32_t acl_model_id;
  auto acl_ret = aclmdlLoadFromMem(om_data.Data(), om_data.DataSize(), &acl_model_id);
  if (acl_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed.";
    return FAILED;
  }

  // acl init model resource
  model_process_.set_model_id(acl_model_id);
  Status ret = model_process_.PreInitModelResource();
  if (ret != SUCCESS) {
    (void)aclmdlUnload(acl_model_id);
    MS_LOG(ERROR) << "Pre init model resource failed.";
    return FAILED;
  }

  MS_LOG(INFO) << "Load acl model success.";
  return SUCCESS;
}

Status AclGraphImpl::InitEnv() {
  if (init_flag_) {
    return SUCCESS;
  }

  acl_env_ = AclEnvGuard::GetAclEnv("");
  if (acl_env_ == nullptr) {
    MS_LOG(ERROR) << "Acl init failed.";
    return FAILED;
  }

  aclError ret = aclrtSetDevice(device_id_);
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

  aclrtRunMode run_mode;
  ret = aclrtGetRunMode(&run_mode);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Acl get run mode failed";
    return FAILED;
  }
  bool is_device = (run_mode == ACL_DEVICE);
  model_process_.SetIsDevice(is_device);
  MS_LOG(INFO) << "Get run mode success is device input/output " << is_device;

  MS_LOG(INFO) << "Init acl success, device id " << device_id_;
  init_flag_ = true;
  return SUCCESS;
}

Status AclGraphImpl::FinalizeEnv() {
  if (!init_flag_) {
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

  if (context_ != nullptr) {
    rt_ret = aclrtDestroyContext(context_);
    if (rt_ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Destroy context failed";
    }
    context_ = nullptr;
  }
  MS_LOG(INFO) << "End to destroy context";

  rt_ret = aclrtResetDevice(device_id_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Reset device " << device_id_ << " failed";
  }
  MS_LOG(INFO) << "End to reset device " << device_id_;

  init_flag_ = false;
  return SUCCESS;
}

Status AclGraphImpl::Load() {
  // check graph type
  if (graph_->ModelType() != ModelType::kOM) {
    Status ret = ConvertToOM();
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Load Failed.";
      return FAILED;
    }
  }

  const auto &graph_data = GraphImpl::MutableGraphData();
  MS_EXCEPTION_IF_NULL(graph_data);
  auto om_data = graph_data->GetOMData();

  // init
  Status ret = InitEnv();
  if (ret != SUCCESS) {
    MS_LOG(ERROR) << "InitEnv failed.";
    return FAILED;
  }

  // load model
  if (!load_flag_) {
    ret = LoadAclModel(om_data);
    if (ret != SUCCESS) {
      MS_LOG(ERROR) << "Load acl model failed.";
      return ret;
    }
    load_flag_ = true;
  }

  aclError rt_ret = aclrtSetCurrentContext(context_);
  if (rt_ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Set the ascend device context failed";
    return FAILED;
  }

  return SUCCESS;
}

Status AclGraphImpl::ConvertToOM() {
  MS_LOG(INFO) << "Start convert to om model.";
  if (graph_ == nullptr) {
    MS_LOG(ERROR) << "Invalid graph_ is null.";
    return FAILED;
  }

  auto &graph_data = GraphImpl::MutableGraphData();
  MS_EXCEPTION_IF_NULL(graph_data);
  if (graph_->ModelType() == ModelType::kOM) {
    MS_LOG(INFO) << "This model has been built, skip.";
    return SUCCESS;
  } else if (graph_->ModelType() == ModelType::kMindIR) {
    auto func_graph = graph_data->GetFuncGraph();
    MS_EXCEPTION_IF_NULL(func_graph);
    ModelConverter model_converter;
    Buffer om_data = model_converter.LoadMindIR(func_graph);
    if (om_data.Data() == nullptr || om_data.DataSize() == 0) {
      MS_LOG(ERROR) << "Convert MindIR to OM failed.";
      return FAILED;
    }
    graph_data = std::make_shared<Graph::GraphData>(om_data, ModelType::kOM);
    MS_LOG(INFO) << "Convert MindIR to OM success.";
    return SUCCESS;
  }
  MS_LOG(ERROR) << "Unsupported ModelType " << graph_->ModelType();
  return FAILED;
}
}  // namespace mindspore::api
