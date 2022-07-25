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

namespace mindspore {
std::mutex g_impl_init_lock;

Model::Model() : impl_(nullptr) {}

Model::~Model() {}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  if (impl_ == nullptr) {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    impl_ = std::make_shared<ModelImpl>();
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

Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  if (impl_ == nullptr) {
    std::unique_lock<std::mutex> impl_lock(g_impl_init_lock);
    impl_ = std::make_shared<ModelImpl>();
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Model implement is null.";
      return kLiteFileError;
    }
  }

  Status ret = impl_->Build(CharToString(model_path), model_type, model_context);
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}

// to do, now just to adapter benchmark
Status Model::Build(const std::vector<char> &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key,
                    const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path) {
  return Build(model_path, model_type, model_context);
}

Status Model::Build(GraphCell graph, const std::shared_ptr<Context> &model_context,
                    const std::shared_ptr<TrainCfg> &train_cfg) {
  MS_LOG(ERROR) << "Unsupported Feature.";
  return kLiteNotSupport;
}

Status Model::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Resize(inputs, dims);
}

Status Model::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Predict(inputs, outputs);
}

Status Model::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                                    const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->PredictWithPreprocess(inputs, outputs);
}

Status Model::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return kMCFailed;
  }
  return impl_->Preprocess(inputs, outputs);
}

bool Model::HasPreprocess() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return false;
  }
  return impl_->HasPreprocess();
}

std::vector<MSTensor> Model::GetInputs() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return {};
  }
  return impl_->GetInputs();
}

std::vector<MSTensor> Model::GetOutputs() {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Failed because this model has not been built.";
    return {};
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
  return std::vector<MSTensor>();
}

Status Model::BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                                    std::map<std::string, unsigned int> *outputGLTexture) {
  return kSuccess;
}

Status Model::LoadConfig(const std::vector<char> &config_path) { return kSuccess; }

Status Model::UpdateConfig(const std::vector<char> &section,
                           const std::pair<std::vector<char>, std::vector<char>> &config) {
  return kSuccess;
}
}  // namespace mindspore
