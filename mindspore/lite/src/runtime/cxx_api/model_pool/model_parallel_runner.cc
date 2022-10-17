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
#include "src/runtime/cxx_api/model_pool/model_parallel_runner_impl.h"
#include "src/runtime/cxx_api/model_pool/runner_config.h"
#include "src/common/log_adapter.h"
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

std::mutex g_model_parallel_runner_mutex;

RunnerConfig::RunnerConfig() : data_(std::make_shared<Data>()) {}

RunnerConfig::~RunnerConfig() {}

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

void RunnerConfig::SetConfigInfo(const std::vector<char> &section,
                                 const std::map<std::vector<char>, std::vector<char>> &config) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return;
  }
  if (data_->config_info.size() > kMaxSectionNum) {
    MS_LOG(ERROR) << "The number of added sessions exceeds the maximum[" << kMaxSectionNum << "] limit.";
    return;
  }
  if (config.size() > kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "The number of added config exceeds the maximum[" << kMaxConfigNumPerSection << "] limit.";
    return;
  }
  data_->config_info[CharToString(section)] = MapVectorCharToString(config);
  return;
}

void RunnerConfig::SetConfigPath(const std::vector<char> &config_path) {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    return;
  }
  data_->config_path = CharToString(config_path);
  return;
}

std::vector<char> RunnerConfig::GetConfigPathChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    std::vector<char> empty;
    return empty;
  }
  return StringToChar(data_->config_path);
}

std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> RunnerConfig::GetConfigInfoChar() const {
  if (data_ == nullptr) {
    MS_LOG(ERROR) << "Runner config data is nullptr.";
    std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> empty;
    return empty;
  }
  return MapMapStringToChar(data_->config_info);
}

ModelParallelRunner::ModelParallelRunner() {}

ModelParallelRunner::~ModelParallelRunner() {}

Status ModelParallelRunner::Init(const std::vector<char> &model_path,
                                 const std::shared_ptr<RunnerConfig> &runner_config) {
  {
    std::lock_guard<std::mutex> l(g_model_parallel_runner_mutex);
    if (model_parallel_runner_impl_ == nullptr) {
#ifdef USE_GLOG
      mindspore::mindspore_log_init();
#endif
      model_parallel_runner_impl_ = std::make_shared<ModelParallelRunnerImpl>();
      if (model_parallel_runner_impl_ == nullptr) {
        MS_LOG(ERROR) << "new model pool failed, model pool is nullptr.";
        return kLiteNullptr;
      }
    }
  }
  return model_parallel_runner_impl_->Init(CharToString(model_path), runner_config);
}

std::vector<MSTensor> ModelParallelRunner::GetInputs() {
  if (model_parallel_runner_impl_ == nullptr) {
    std::vector<MSTensor> empty;
    MS_LOG(ERROR) << "Please initialize ModelParallelRunner before calling GetInput API.";
    return empty;
  }
  return model_parallel_runner_impl_->GetInputs();
}

std::vector<MSTensor> ModelParallelRunner::GetOutputs() {
  if (model_parallel_runner_impl_ == nullptr) {
    std::vector<MSTensor> empty;
    MS_LOG(ERROR) << "Please initialize ModelParallelRunner before calling GetOutputs API.";
    return empty;
  }
  return model_parallel_runner_impl_->GetOutputs();
}

Status ModelParallelRunner::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (model_parallel_runner_impl_ == nullptr) {
    MS_LOG(ERROR) << "ModelParallelRunner Not Initialize.";
    return kLiteNullptr;
  }
  return model_parallel_runner_impl_->Predict(inputs, outputs, before, after);
}
}  // namespace mindspore
