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
#include "include/api/model.h"
#include "include/api/context.h"
#include "extendrt/cxx_api/model/model_impl.h"
#ifdef ENABLE_OPENSSL
#include "src/common/decrypt.h"
#include "src/common/file_utils.h"
#endif

namespace mindspore {
namespace {
#ifdef USE_GLOG
extern "C" {
extern void mindspore_log_init();
}
#endif
}  // namespace

Model::Model() {
#ifdef USE_GLOG
#if defined(_WIN32) || defined(_WIN64) || defined(__APPLE__)
#ifdef _MSC_VER
  mindspore::mindspore_log_init();
#endif
#else
  mindspore::mindspore_log_init();
#endif
#endif
  impl_ = std::make_shared<ModelImpl>();
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create ModelImpl";
  }
}

Model::~Model() {}

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
                    const std::shared_ptr<Context> &model_context) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  try {
    Status ret = impl_->Build(model_data, data_size, model_type, model_context);
    if (ret != kSuccess) {
      return ret;
    }
    return kSuccess;
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return kCoreFailed;
  }
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  try {
    Status ret = impl_->Build(CharToString(model_path), model_type, model_context);
    if (ret != kSuccess) {
      return ret;
    }
    return kSuccess;
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return kCoreFailed;
  }
}

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
#ifdef ENABLE_OPENSSL
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }

  if (dec_key.len > 0) {
    size_t model_size;
    auto model_buf = lite::ReadFile(model_path.data(), &model_size);
    if (model_buf == nullptr) {
      MS_LOG(ERROR) << "Read model file failed";
      return kLiteFileError;
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
    try {
      ret = impl_->Build(decrypt_buffer.get(), decrypt_len, model_type, model_context);
      if (ret != kSuccess) {
        delete[] model_buf;
        return ret;
      }
      delete[] model_buf;
      return kSuccess;
    } catch (const std::exception &exe) {
      delete[] model_buf;
      MS_LOG_ERROR << "Catch exception: " << exe.what();
      return kCoreFailed;
    }
  } else {
    try {
      Status ret = impl_->Build(CharToString(model_path), model_type, model_context);
      if (ret != kSuccess) {
        return ret;
      }
      return kSuccess;
    } catch (const std::exception &exe) {
      MS_LOG_ERROR << "Catch exception: " << exe.what();
      return kCoreFailed;
    }
  }
#else
  MS_LOG(ERROR) << "The lib is not support Decrypt Model.";
  return kLiteError;
#endif
}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
#ifdef ENABLE_OPENSSL
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
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
    try {
      ret = impl_->Build(decrypt_buffer.get(), decrypt_len, model_type, model_context);
      if (ret != kSuccess) {
        return ret;
      }
      return kSuccess;
    } catch (const std::exception &exe) {
      MS_LOG_ERROR << "Catch exception: " << exe.what();
      return kCoreFailed;
    }
  } else {
    try {
      Status ret = impl_->Build(model_data, data_size, model_type, model_context);
      if (ret != kSuccess) {
        return ret;
      }
      return kSuccess;
    } catch (const std::exception &exe) {
      MS_LOG_ERROR << "Catch exception: " << exe.what();
      return kCoreFailed;
    }
  }
#else
  MS_LOG(ERROR) << "The lib is not support Decrypt Model.";
  return kLiteError;
#endif
}

Status Model::Build(GraphCell graph, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &train_cfg) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Build(GraphCell graph, Node *optimizer, std::vector<Expr *> inputs,
                    const std::shared_ptr<Context> &model_context, const std::shared_ptr<TrainCfg> &train_cfg) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context,
                             const std::shared_ptr<TrainCfg> &train_cfg = nullptr) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  try {
    return impl_->Resize(inputs, dims);
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return kCoreFailed;
  }
}

Status Model::RunStep(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  auto inputs = impl_->GetInputs();
  auto outputs = impl_->GetOutputs();
  return impl_->Predict(inputs, &outputs);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return kLiteNullptr;
  }
  try {
    return impl_->Predict(inputs, outputs, before, after);
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return kCoreFailed;
  }
}

Status Model::Predict(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
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

std::vector<MSTensor> Model::GetInputs() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return {};
  }
  try {
    return impl_->GetInputs();
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return {};
  }
}

std::vector<MSTensor> Model::GetOutputs() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return {};
  }
  try {
    return impl_->GetOutputs();
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return {};
  }
}

MSTensor Model::GetInputByTensorName(const std::vector<char> &name) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return MSTensor(nullptr);
  }
  try {
    return impl_->GetInputByTensorName(CharToString(name));
  } catch (const std::exception &exe) {
    MS_LOG_ERROR << "Catch exception: " << exe.what();
    return {};
  }
}

std::vector<std::vector<char>> Model::GetOutputTensorNamesChar() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return {};
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
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                    std::map<std::string, unsigned int> *outputGLTexture) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::LoadConfig(const std::vector<char> &config_path) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteFileError, "Fail to load config file.");
  }

  auto ret = impl_->LoadConfig(CharToString(config_path));
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Fail to load config file.";
    return Status(kLiteFileError, "Invalid config file.");
  }
  return kSuccess;
}

Status Model::UpdateConfig(const std::vector<char> &section,
                           const std::pair<std::vector<char>, std::vector<char>> &config) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Model implement is null.";
    return Status(kLiteFileError, "Fail to update config file.");
  }
  auto ret = impl_->UpdateConfig(CharToString(section), {CharToString(config.first), CharToString(config.second)});
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Fail to update config file.";
    return Status(kLiteFileError, "Fail to update config file.");
  }
  return kSuccess;
}

bool Model::CheckModelSupport(enum DeviceType device_type, ModelType model_type) {
  return ModelImpl::CheckModelSupport(device_type, model_type);
}

Status Model::UpdateWeights(const std::vector<MSTensor> &new_weights) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<MSTensor> Model::GetTrainableParams() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

std::vector<MSTensor> Model::GetGradients() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::ApplyGradients(const std::vector<MSTensor> &gradients) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<MSTensor> Model::GetFeatureMaps() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::UpdateFeatureMaps(const std::vector<MSTensor> &new_weights) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<MSTensor> Model::GetOptimizerParams() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::SetOptimizerParams(const std::vector<MSTensor> &params) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::SetLearningRate(float learning_rate) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

float Model::GetLearningRate() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return 0.0;
}

Status Model::InitMetrics(const std::vector<Metrics *> metrics) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

std::vector<Metrics *> Model::GetMetrics() {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return {};
}

Status Model::SetTrainMode(bool train) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

bool Model::GetTrainMode() const {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return false;
}

// cppcheck-suppress passedByValue
Status Train(int epochs, std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> cbs) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

// cppcheck-suppress passedByValue
Status Evaluate(std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> cbs) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}
}  // namespace mindspore
