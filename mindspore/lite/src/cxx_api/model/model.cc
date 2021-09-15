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

#include "include/api/model.h"
#include <mutex>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/callback/callback.h"
#include "include/api/dual_abi_helper.h"
#include "src/cxx_api/model/model_impl.h"
#include "src/cxx_api/callback/callback_impl.h"
#include "src/cxx_api/callback/callback_adapter.h"
#include "src/common/log_adapter.h"

namespace mindspore {
std::mutex g_impl_init_lock;

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::string &dec_mode) {
  if (impl_ == nullptr) {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    impl_ = std::shared_ptr<ModelImpl>(new (std::nothrow) ModelImpl());
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Model implement is null.";
      return kLiteFileError;
    }
  }

  Status ret = impl_->Build(model_data, data_size, model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

Status Model::Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context,
                    const Key &dec_key, const std::string &dec_mode) {
  if (impl_ == nullptr) {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    impl_ = std::shared_ptr<ModelImpl>(new (std::nothrow) ModelImpl());
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Model implement is null.";
      return kLiteFileError;
    }
  }

  Status ret = impl_->Build(model_path, model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

Status Model::Build(GraphCell graph, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &train_cfg) {
  std::stringstream err_msg;
  if (impl_ == nullptr) {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    impl_ = std::shared_ptr<ModelImpl>(new (std::nothrow) ModelImpl());
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Model implement is null.";
      return kLiteFileError;
    }
  }

  if (graph.GetGraph() == nullptr) {
    err_msg << "Invalid null graph.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  if (model_context == nullptr) {
    err_msg << "Invalid null context.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  impl_->SetContext(model_context);
  impl_->SetGraph(graph.GetGraph());
  impl_->SetConfig(train_cfg);
  return impl_->Build();
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Predict(inputs, outputs, before, after);
}

Status Model::PredictWithPreprocess(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Preprocess(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

bool Model::HasPreprocess() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return false;
}

Model::Model() : impl_(nullptr) {}

Model::~Model() {}

bool Model::CheckModelSupport(enum DeviceType device_type, ModelType model_type) {
  MS_LOG(ERROR) << "Unsupported feature.";
  return false;
}

std::vector<MSTensor> Model::GetInputs() {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetInputs();
}

std::vector<MSTensor> Model::GetOutputs() {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetOutputs();
}

MSTensor Model::GetInputByTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return MSTensor(nullptr);
  }
  return impl_->GetInputByTensorName(CharToString(name));
}

std::vector<std::vector<char>> Model::GetOutputTensorNamesChar() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    std::vector<std::vector<char>> empty;
    return empty;
  }
  return VectorStringToChar(impl_->GetOutputTensorNames());
}

MSTensor Model::GetOutputByTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return MSTensor(nullptr);
  }
  return impl_->GetOutputByTensorName(CharToString(name));
}

std::vector<MSTensor> Model::GetOutputsByNodeName(const std::vector<char> &node_name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    std::vector<MSTensor> empty;
    return empty;
  }
  return impl_->GetOutputsByNodeName(CharToString(node_name));
}

Status Model::LoadConfig(const std::string &config_path) {
  std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
  if (impl_ != nullptr) {
    MS_LOG(ERROR) << "impl_ illegal in LoadConfig.";
    return Status(kLiteFileError, "Illegal operation.");
  }

  impl_ = std::shared_ptr<ModelImpl>(new (std::nothrow) ModelImpl());
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteFileError, "Fail to load config file.");
  }

  auto ret = impl_->LoadConfig(config_path);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "impl_ LoadConfig failed,";
    return Status(kLiteFileError, "Invalid config file.");
  }
  return kSuccess;
}

Status Model::SetTrainMode(bool train) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  auto ret = (train) ? impl_->session_->Train() : impl_->session_->Eval();
  return (ret == mindspore::lite::RET_OK) ? kSuccess : kLiteError;
}

bool Model::GetTrainMode() const { return ((impl_ != nullptr) && (impl_->session_) && (impl_->session_->IsTrain())); }

std::vector<MSTensor> Model::GetGradients() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetGradients();
}

Status Model::ApplyGradients(const std::vector<MSTensor> &gradients) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  return impl_->ApplyGradients(gradients);
}

std::vector<MSTensor> Model::GetOptimizerParams() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  auto res = impl_->GetOptimizerParams();
  return res;
}

Status Model::SetOptimizerParams(const std::vector<MSTensor> &params) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  return impl_->SetOptimizerParams(params);
}

Status Model::InitMetrics(std::vector<Metrics *> metrics) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->InitMetrics(metrics);
}

std::vector<Metrics *> Model::GetMetrics() {
  std::vector<Metrics *> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetMetrics();
}

}  // namespace mindspore
