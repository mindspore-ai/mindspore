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
#ifndef MINDSPORE_INCLUDE_API_MODEL_RUNNER_H
#define MINDSPORE_INCLUDE_API_MODEL_RUNNER_H
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include "include/api/status.h"
#include "include/api/context.h"

namespace mindspore {
class ModelPool;

struct RunnerConfig {
  RunnerConfig(std::shared_ptr<Context> &ctx, int num) : model_ctx(ctx), num_model(num) {}
  std::shared_ptr<Context> model_ctx = nullptr;
  int num_model;
};

/// \brief The ModelRunner class is used to define a MindSpore ModelPoolManager, facilitating Model management.
class MS_API ModelParallelRunner {
 public:
  ModelParallelRunner() = default;
  ~ModelParallelRunner() = default;

  /// \brief build a model runner from model path so that it can run on a device. Only valid for Lite.
  ///
  /// \param[in] model_path Define the model path.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  /// \param[in] dec_key Define the key used to decrypt the ciphertext model. The key length is 16, 24, or 32.
  /// \param[in] dec_mode Define the decryption mode. Options: AES-GCM, AES-CBC.
  ///
  /// \return Status.
  Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr,
              const Key &dec_key = {}, const std::string &dec_mode = kDecModeAesGcm);

  /// \brief Obtains all input tensors of the model.
  ///
  /// \return The vector that includes all input tensors.
  std::vector<MSTensor> GetInputs();

  /// \brief Inference ModelPoolManager.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_RUNNER_H
