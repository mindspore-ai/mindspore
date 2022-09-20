/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/api/model_parallel_runner.h"
#include "src/runtime/cxx_api/model_pool/model_pool.h"
#include "src/runtime/cxx_api/model_pool/runner_config.h"
#include "src/common/log_adapter.h"
#include "src/runtime/cpu_info.h"
#ifdef CAPTURE_SIGNALS
#include "src/runtime/signal_handler.h"
#endif
namespace mindspore {
namespace {
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
}  // namespace
#ifdef USE_GLOG
extern "C" {
extern void mindspore_log_init();
}
#endif

RunnerConfig::RunnerConfig() : data_(std::make_shared<Data>()) {}

void RunnerConfig::SetWorkersNum(int32_t workers_num) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return;
  }
  data_->workers_num = workers_num;
}

void RunnerConfig::SetContext(const std::shared_ptr<Context> &context) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return;
  }
  data_->context = context;
}

int32_t RunnerConfig::GetWorkersNum() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return -1;
  }
  return data_->workers_num;
}

std::shared_ptr<Context> RunnerConfig::GetContext() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return nullptr;
  }
  return data_->context;
}

void RunnerConfig::SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return;
  }
  if (data_->config_info.size() > kMaxSectionNum) {
    return;
  }
  if (config.size() > kMaxConfigNumPerSection) {
    return;
  }
  data_->config_info[section] = config;
  return;
}

std::map<std::string, std::map<std::string, std::string>> RunnerConfig::GetConfigInfo() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    std::map<std::string, std::map<std::string, std::string>> empty;
    return empty;
  }
  return data_->config_info;
}

void RunnerConfig::SetConfigPath(const std::string &config_path) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return;
  }
  data_->config_path = config_path;
  return;
}

std::string RunnerConfig::GetConfigPath() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    std::string empty;
    return empty;
  }
  return data_->config_path;
}

Status ModelParallelRunner::Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config) {
#ifdef USE_GLOG
  mindspore::mindspore_log_init();
#endif
  if (model_pool_ != nullptr && model_pool_->IsInitialized()) {
    MS_LOG(WARNING) << "ModelParallelRunner is already initialized, not need to initialize it again";
    return kSuccess;
  }
  auto new_model_pool = std::make_shared<ModelPool>();
  if (new_model_pool == nullptr) {
    MS_LOG(ERROR) << "new model pool failed, model pool is nullptr.";
    return kLiteNullptr;
  }
  if (!PlatformInstructionSetSupportCheck()) {
    return kLiteNotSupport;
  }
  auto status = new_model_pool->Init(model_path, runner_config);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "ModelParallelRunner init failed.";
    return kLiteError;
  }
  if (model_pool_ != nullptr && model_pool_->IsInitialized()) {
    MS_LOG(WARNING) << "ModelParallelRunner is already initialized, not need to initialize it again";
    return kSuccess;
  }
  model_pool_ = new_model_pool;
#ifdef CAPTURE_SIGNALS
  CaptureSignal();
#endif
  return status;
}

std::vector<MSTensor> ModelParallelRunner::GetInputs() {
  if (model_pool_ == nullptr) {
    std::vector<MSTensor> empty;
    MS_LOG(ERROR) << "Please initialize ModelParallelRunner before calling GetInput API.";
    return empty;
  }
  return model_pool_->GetInputs();
}

std::vector<MSTensor> ModelParallelRunner::GetOutputs() {
  if (model_pool_ == nullptr) {
    std::vector<MSTensor> empty;
    MS_LOG(ERROR) << "Please initialize ModelParallelRunner before calling GetInput API.";
    return empty;
  }
  return model_pool_->GetOutputs();
}

Status ModelParallelRunner::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (outputs == nullptr || model_pool_ == nullptr) {
    MS_LOG(ERROR) << "predict output is nullptr or ModelParallelRunner Not Initialize.";
    return kLiteNullptr;
  }
  auto status = model_pool_->Predict(inputs, outputs, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "ModelParallelRunner predict failed.";
    return status;
  }
  return kSuccess;
}
}  // namespace mindspore
