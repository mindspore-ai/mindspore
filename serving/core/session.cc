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
#include "core/session.h"
#include <grpcpp/grpcpp.h>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <chrono>

#include "include/infer_log.h"
#include "serving/ms_service.grpc.pb.h"
#include "core/util/option_parser.h"
#include "core/version_control/version_controller.h"
#include "core/util/file_system_operation.h"
#include "core/serving_tensor.h"

using ms_serving::MSService;
using ms_serving::PredictReply;
using ms_serving::PredictRequest;

namespace mindspore {
namespace serving {

Status Session::CreatDeviceSession(const std::string &device, uint32_t device_id) {
  session_ = inference::InferSession::CreateSession(device, device_id);
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "Creat Session Failed";
    return FAILED;
  }
  device_type_ = device;
  return SUCCESS;
}

Session &Session::Instance() {
  static Session instance;
  return instance;
}

Status Session::Predict(const PredictRequest &request, PredictReply &reply) {
  if (!model_loaded_) {
    MSI_LOG(ERROR) << "the model has not loaded";
    return FAILED;
  }
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "the inference session has not be initialized";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  MSI_LOG(INFO) << "run Predict";

  if (request.images_size() > 0) {
    ServingImagesRequest serving_images(request);
    ServingRequest serving_request(request);
    ServingReply serving_reply(reply);
    Status ret = session_->ExecuteModel(graph_id_, serving_images, serving_request, serving_reply);
    if (ret != SUCCESS) {
      MSI_LOG(ERROR) << "execute model with images return failed";
      return ret;
    }
  } else if (request.data_size() > 0) {
    ServingRequest serving_request(request);
    ServingReply serving_reply(reply);
    Status ret = session_->ExecuteModel(graph_id_, serving_request, serving_reply);
    if (ret != SUCCESS) {
      MSI_LOG(ERROR) << "execute model with datas return failed";
      return ret;
    }
  }

  MSI_LOG(INFO) << "run Predict finished";
  return SUCCESS;
}

Status Session::Warmup(const MindSporeModelPtr model) {
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "The CreatDeviceSession should be called, before warmup";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  std::string file_name = model->GetModelPath() + '/' + model->GetModelName();
  model_loaded_ = false;
  MSI_TIME_STAMP_START(LoadModelFromFile)
  auto ret = session_->LoadModelFromFile(file_name, graph_id_);
  MSI_TIME_STAMP_END(LoadModelFromFile)
  if (ret != SUCCESS) {
    MSI_LOG(ERROR) << "Load graph model failed, file name is " << file_name.c_str();
    return ret;
  }
  model_loaded_ = true;
  MSI_LOG(INFO) << "Session Warmup finished";
  return SUCCESS;
}

Status Session::Clear() {
  if (session_ != nullptr) {
    session_->UnloadModel(graph_id_);
    session_->FinalizeEnv();
    session_ = nullptr;
  }
  return SUCCESS;
}

Status Session::GetModelInputsInfo(std::vector<inference::InferTensor> &tensor_list) {
  if (!model_loaded_) {
    MSI_LOG(ERROR) << "the model has not loaded";
    return FAILED;
  }
  if (session_ == nullptr) {
    MSI_LOG(ERROR) << "the inference session has not be initialized";
    return FAILED;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Status ret = session_->GetModelInputsInfo(graph_id_, &tensor_list);
  if (ret != SUCCESS) {
    MSI_LOG(ERROR) << "get model inputs info failed";
  }
  return ret;
}

}  // namespace serving
}  // namespace mindspore
