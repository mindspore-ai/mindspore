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
#include "src/extendrt/cxx_api/model_pool/model_parallel_runner_impl.h"
#include "src/extendrt/cxx_api/model_pool/runner_config.h"
#include "src/common/log_adapter.h"
#include "src/litert/cpu_info.h"
#include "nnacl/op_base.h"
#ifdef CAPTURE_SIGNALS
#include "src/extendrt/signal_handler.h"
#endif
namespace mindspore {
Status ModelParallelRunnerImpl::Init(const std::string &model_path,
                                     const std::shared_ptr<RunnerConfig> &runner_config) {
  std::unique_lock<std::shared_mutex> l(model_parallel_runner_impl_mutex_);
  if (model_pool_ != nullptr) {
    MS_LOG(WARNING) << "ModelParallelRunner is already initialized, not need to initialize it again";
    return kSuccess;
  }
  model_pool_ = new (std::nothrow) ModelPool();
  if (model_pool_ == nullptr) {
    MS_LOG(ERROR) << "new model pool failed, model pool is nullptr.";
    return kLiteNullptr;
  }
  auto status = model_pool_->InitByPath(model_path, runner_config);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "ModelParallelRunner init failed.";
    delete model_pool_;
    model_pool_ = nullptr;
    return kLiteError;
  }
#ifdef CAPTURE_SIGNALS
  CaptureSignal();
#endif
  return status;
}

Status ModelParallelRunnerImpl::Init(const void *model_data, size_t data_size,
                                     const std::shared_ptr<RunnerConfig> &runner_config) {
  std::unique_lock<std::shared_mutex> l(model_parallel_runner_impl_mutex_);
  if (model_pool_ != nullptr) {
    MS_LOG(WARNING) << "ModelParallelRunner is already initialized, not need to initialize it again";
    return kSuccess;
  }
  model_pool_ = new (std::nothrow) ModelPool();
  if (model_pool_ == nullptr) {
    MS_LOG(ERROR) << "new model pool failed, model pool is nullptr.";
    return kLiteNullptr;
  }
  auto status = model_pool_->InitByBuf(static_cast<const char *>(model_data), data_size, runner_config);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "ModelParallelRunner init failed.";
    delete model_pool_;
    model_pool_ = nullptr;
    return kLiteError;
  }
#ifdef CAPTURE_SIGNALS
  CaptureSignal();
#endif
  return status;
}

std::vector<MSTensor> ModelParallelRunnerImpl::GetInputs() {
  std::shared_lock<std::shared_mutex> l(model_parallel_runner_impl_mutex_);
  if (model_pool_ == nullptr) {
    std::vector<MSTensor> empty;
    MS_LOG(ERROR) << "Please initialize ModelParallelRunner before calling GetInput API.";
    return empty;
  }
  return model_pool_->GetInputs();
}

std::vector<MSTensor> ModelParallelRunnerImpl::GetOutputs() {
  std::shared_lock<std::shared_mutex> l(model_parallel_runner_impl_mutex_);
  if (model_pool_ == nullptr) {
    std::vector<MSTensor> empty;
    MS_LOG(ERROR) << "Please initialize ModelParallelRunner before calling GetOutputs API.";
    return empty;
  }
  return model_pool_->GetOutputs();
}

Status ModelParallelRunnerImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                                        const MSKernelCallBack &before, const MSKernelCallBack &after) {
  std::shared_lock<std::shared_mutex> l(model_parallel_runner_impl_mutex_);
  if (MS_UNLIKELY((outputs == nullptr || model_pool_ == nullptr))) {
    MS_LOG(ERROR) << "predict output is nullptr or ModelParallelRunner Not Initialize.";
    return kLiteNullptr;
  }
  auto status = model_pool_->Predict(inputs, outputs, before, after);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "model runner predict failed.";
    return status;
  }
  return kSuccess;
}
ModelParallelRunnerImpl::~ModelParallelRunnerImpl() {
  MS_LOG(INFO) << "delete model pool begin.";
  std::unique_lock<std::shared_mutex> l(model_parallel_runner_impl_mutex_);
  if (model_pool_ != nullptr) {
    MS_LOG(INFO) << "delete model pool impl.";
    delete model_pool_;
    model_pool_ = nullptr;
  } else {
    MS_LOG(INFO) << "model pool is nullptr.";
  }
  MS_LOG(INFO) << "delete model pool done.";
}
}  // namespace mindspore
