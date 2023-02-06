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
#ifdef GPU_TENSORRT
#include <cuda_runtime.h>
#endif
#include "flatbuffers/flatbuffers.h"
#include "include/api/callback/callback.h"
#include "include/api/context.h"
#include "include/api/dual_abi_helper.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/api/graph.h"
#include "src/common/log_adapter.h"
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
#include "src/common/thread_utils.h"
#endif
#include "src/litert/cxx_api/expression/net_impl.h"
#include "src/litert/cxx_api/callback/callback_adapter.h"
#include "src/litert/cxx_api/callback/callback_impl.h"
#include "src/litert/cxx_api/model/model_impl.h"
#ifdef ENABLE_OPENSSL
#include "src/common/decrypt.h"
#include "src/common/file_utils.h"
#endif

namespace mindspore {
#ifdef USE_GLOG
extern "C" {
extern void mindspore_log_init();
}
#endif

std::mutex g_impl_init_lock;
#ifdef ENABLE_OPENSSL
Status DecryptModel(const std::string &cropto_lib_path, const void *model_buf, size_t model_size, const Key &dec_key,
                    const std::string &dec_mode, std::unique_ptr<Byte[]> *decrypt_buffer, size_t *decrypt_len) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "model_buf is nullptr.";
    return kLiteError;
  }
  *decrypt_len = 0;
  *decrypt_buffer = lite::Decrypt(cropto_lib_path, decrypt_len, reinterpret_cast<const Byte *>(model_buf), model_size,
                                  dec_key.key, dec_key.len, dec_mode);
  if (*decrypt_buffer == nullptr || *decrypt_len == 0) {
    MS_LOG(ERROR) << "Decrypt buffer failed";
    return kLiteError;
  }
  return kSuccess;
}
#endif

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
#ifdef ENABLE_OPENSSL
  {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    if (impl_ == nullptr) {
#ifdef USE_GLOG
      mindspore::mindspore_log_init();
#endif
      impl_ = std::make_shared<ModelImpl>();
      if (impl_ == nullptr) {
        MS_LOG(ERROR) << "Model implement is null.";
        return kLiteFileError;
      }
    }
  }
  if (dec_key.len > 0) {
    std::unique_ptr<Byte[]> decrypt_buffer;
    size_t decrypt_len = 0;
    Status ret = DecryptModel(CharToString(cropto_lib_path), model_data, data_size, dec_key, CharToString(dec_mode),
                              &decrypt_buffer, &decrypt_len);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Decrypt model failed.";
      return ret;
    }
    ret = impl_->Build(decrypt_buffer.get(), decrypt_len, model_type, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed.";
      return ret;
    }
  } else {
    Status ret;
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
    if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
      pid_t pid = fork();
      if (pid < 0) {
        return kLiteError;
      } else if (pid == 0) {  // child process
        ret = impl_->BuildAndRun(model_data, data_size, model_type, model_context);
        int ret_code = ret == kSuccess ? lite::kProcessSuccess : lite::kProcessFailed;
        exit(ret_code);
      }
      ret = lite::CheckPidStatus(pid);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "PreBuild or PreInference failed.";
        return ret;
      }
    }
#endif
    ret = impl_->Build(model_data, data_size, model_type, model_context);
    if (ret != kSuccess) {
      return ret;
    }
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "The lib is not support Decrypt Model.";
  return kLiteError;
#endif
}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    if (impl_ == nullptr) {
#ifdef USE_GLOG
      mindspore::mindspore_log_init();
#endif
      impl_ = std::make_shared<ModelImpl>();
      if (impl_ == nullptr) {
        MS_LOG(ERROR) << "Model implement is null.";
        return kLiteFileError;
      }
    }
  }
  Status ret;
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {  // child process
      ret = impl_->BuildAndRun(model_data, data_size, model_type, model_context);
      int ret_code = ret == kSuccess ? lite::kProcessSuccess : lite::kProcessFailed;
      exit(ret_code);
    }
    ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreBuild or PreInference failed.";
      return ret;
    }
  }
#endif
  ret = impl_->Build(model_data, data_size, model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
#ifdef ENABLE_OPENSSL
  {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    if (impl_ == nullptr) {
#ifdef USE_GLOG
      mindspore::mindspore_log_init();
#endif
      impl_ = std::make_shared<ModelImpl>();
      if (impl_ == nullptr) {
        MS_LOG(ERROR) << "Model implement is null.";
        return kLiteFileError;
      }
    }
  }
  if (dec_key.len > 0) {
    size_t model_size;
    auto model_buf = lite::ReadFile(CharToString(model_path).c_str(), &model_size);
    if (model_buf == nullptr) {
      MS_LOG(ERROR) << "Read model file failed";
      return kLiteError;
    }
    std::unique_ptr<Byte[]> decrypt_buffer;
    size_t decrypt_len = 0;
    Status ret = DecryptModel(CharToString(cropto_lib_path), model_buf, model_size, dec_key, CharToString(dec_mode),
                              &decrypt_buffer, &decrypt_len);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Decrypt model failed.";
      delete[] model_buf;
      return ret;
    }
    ret = impl_->Build(decrypt_buffer.get(), decrypt_len, model_type, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed.";
      delete[] model_buf;
      return ret;
    }
    delete[] model_buf;
  } else {
    Status ret;
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
    if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
      pid_t pid = fork();
      if (pid < 0) {
        return kLiteError;
      } else if (pid == 0) {  // child process
        ret = impl_->BuildAndRun(CharToString(model_path), model_type, model_context);
        int ret_code = ret == kSuccess ? lite::kProcessSuccess : lite::kProcessFailed;
        exit(ret_code);
      }
      ret = lite::CheckPidStatus(pid);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "PreBuild or PreInference failed.";
        return ret;
      }
    }
#endif
    ret = impl_->Build(CharToString(model_path), model_type, model_context);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Build model failed.";
      return ret;
    }
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "The lib is not support Decrypt Model.";
  return kLiteError;
#endif
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    if (impl_ == nullptr) {
#ifdef USE_GLOG
      mindspore::mindspore_log_init();
#endif
      impl_ = std::make_shared<ModelImpl>();
      if (impl_ == nullptr) {
        MS_LOG(ERROR) << "Model implement is null.";
        return kLiteFileError;
      }
    }
  }
  Status ret;
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {  // child process
      ret = impl_->BuildAndRun(CharToString(model_path), model_type, model_context);
      int ret_code = ret == kSuccess ? lite::kProcessSuccess : lite::kProcessFailed;
      exit(ret_code);
    }
    ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreBuild or PreInference failed.";
      return ret;
    }
  }
#endif
  ret = impl_->Build(CharToString(model_path), model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

Status Model::Build(GraphCell graph, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &train_cfg) {
  std::stringstream err_msg;
  {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    if (impl_ == nullptr) {
#ifdef USE_GLOG
      mindspore::mindspore_log_init();
#endif
      impl_ = std::make_shared<ModelImpl>();
      if (impl_ == nullptr) {
        MS_LOG(ERROR) << "Model implement is null.";
        return kLiteFileError;
      }
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
#if defined(ENABLE_PRE_INFERENCE) && defined(__linux__) && !defined(Debug)
  if (lite::GetNumThreads() == lite::kSingleThread && impl_->IsEnablePreInference()) {
    pid_t pid = fork();
    if (pid < 0) {
      return kLiteError;
    } else if (pid == 0) {  // child process
      auto ret = impl_->BuildAndRun();
      int ret_code = ret == kSuccess ? lite::kProcessSuccess : lite::kProcessFailed;
      exit(ret_code);
    }
    auto ret = lite::CheckPidStatus(pid);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "PreResize or PreInference failed.";
      return ret;
    }
  }
#endif
  return impl_->Build();
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::UpdateWeights(const std::vector<MSTensor> &new_weights) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->UpdateWeights(new_weights);
}

Status Model::RunStep(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  auto inputs = impl_->GetInputs();
  auto outputs = impl_->GetOutputs();
  return impl_->Predict(inputs, &outputs, before, after);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Predict(inputs, outputs, before, after);
}

Status Model::Predict(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  return impl_->Predict(before, after);
}

Status Model::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
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
  if (device_type == kCPU) {
    return true;
  }
#ifdef GPU_TENSORRT
  if (device_type == kGPU) {
    int driver_version = 0;
    int ret = cudaDriverGetVersion(&driver_version);
    if (ret != cudaSuccess || driver_version == 0) {
      MS_LOG(ERROR) << "No nvidia GPU driver.";
      return false;
    }
    return true;
  }
#endif
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

Status Model::BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                    std::map<std::string, unsigned int> *outputGLTexture) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteError;
  }
  return impl_->BindGLTexture2DMemory(inputGLTexture, outputGLTexture);
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

Status Model::LoadConfig(const std::vector<char> &config_path) {
  std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
  if (impl_ != nullptr) {
    MS_LOG(ERROR) << "impl_ illegal in LoadConfig.";
    return Status(kLiteFileError, "Illegal operation.");
  }

  impl_ = std::make_shared<ModelImpl>();
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteFileError, "Fail to load config file.");
  }

  auto ret = impl_->LoadConfig(CharToString(config_path));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "impl_ LoadConfig failed,";
    return Status(kLiteFileError, "Invalid config file.");
  }
  return kSuccess;
}

Status Model::UpdateConfig(const std::vector<char> &section,
                           const std::pair<std::vector<char>, std::vector<char>> &config) {
  std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
  if (impl_ == nullptr) {
    impl_ = std::make_shared<ModelImpl>();
  }
  if (impl_ != nullptr) {
    return impl_->UpdateConfig(CharToString(section), {CharToString(config.first), CharToString(config.second)});
  }
  MS_LOG(ERROR) << "Model implement is null!";
  return kLiteFileError;
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

std::vector<MSTensor> Model::GetFeatureMaps() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetFeatureMaps();
}

std::vector<MSTensor> Model::GetTrainableParams() const {
  std::vector<MSTensor> empty;
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return empty;
  }
  return impl_->GetTrainableParams();
}

Status Model::UpdateFeatureMaps(const std::vector<MSTensor> &new_weights) {
  if ((impl_ == nullptr) || (impl_->session_ == nullptr)) {
    MS_LOG(ERROR) << "Model is null.";
    return kLiteUninitializedObj;
  }
  return impl_->UpdateFeatureMaps(new_weights);
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

Status Model::InitMetrics(const std::vector<Metrics *> metrics) {
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

Status Model::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->SetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
}

Status Model::SetLearningRate(float learning_rate) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteUninitializedObj;
  }
  return impl_->SetLearningRate(learning_rate);
}

float Model::GetLearningRate() {
  if (impl_ == nullptr) {
    MS_LOG(WARNING) << "Model implement is null.";
    return 0.0;
  }
  return impl_->GetLearningRate();
}
}  // namespace mindspore
