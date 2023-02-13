/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_INCLUDE_API_MODEL_PARALLEL_RUNNER_H
#define MINDSPORE_INCLUDE_API_MODEL_PARALLEL_RUNNER_H
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>
#include "include/api/status.h"
#include "include/api/context.h"
namespace mindspore {
/// \brief The RunnerConfig class  is used to store environment variables during execution
/// management.
class MS_API RunnerConfig {
 public:
  struct Data;
  RunnerConfig();
  ~RunnerConfig();

  /// \brief Set the number of workers at runtime. Only valid for ModelParallelRunner.
  ///
  /// \param[in] workers_num the number of workers at runtime.
  void SetWorkersNum(int32_t workers_num);

  /// \brief Get the current operators parallel workers number setting. Only valid for ModelParallelRunner.
  ///
  /// \return The current operators parallel workers number setting.
  int32_t GetWorkersNum() const;

  /// \brief Set the context at runtime. Only valid for ModelParallelRunner.
  ///
  /// \param[in] context store environment variables at runtime.
  void SetContext(const std::shared_ptr<Context> &context);

  /// \brief Get the current context setting. Only valid for ModelParallelRunner.
  ///
  /// \return The current operators context setting.
  std::shared_ptr<Context> GetContext() const;

  /// \brief Set the config before runtime. Only valid for ModelParallelRunner.
  ///
  /// \param[in] section The category of the configuration parameter.
  /// \param[in] config store environment variables before runtime.
  inline void SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config);

  /// \brief Get the current config setting. Only valid for ModelParallelRunner.
  ///
  /// \return The current config setting.
  inline std::map<std::string, std::map<std::string, std::string>> GetConfigInfo() const;

  /// \brief Set the config path before runtime. Only valid for ModelParallelRunner.
  ///
  /// \param[in] config_path The path of the configuration parameter.
  inline void SetConfigPath(const std::string &config_path);

  /// \brief Get the current config path. Only valid for ModelParallelRunner.
  ///
  /// \return The current config path.
  inline std::string GetConfigPath() const;

 private:
  void SetConfigInfo(const std::vector<char> &section, const std::map<std::vector<char>, std::vector<char>> &config);
  std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> GetConfigInfoChar() const;
  void SetConfigPath(const std::vector<char> &config_path);
  std::vector<char> GetConfigPathChar() const;
  std::shared_ptr<Data> data_ = nullptr;
};

void RunnerConfig::SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config) {
  SetConfigInfo(StringToChar(section), MapStringToVectorChar(config));
}

std::map<std::string, std::map<std::string, std::string>> RunnerConfig::GetConfigInfo() const {
  return MapMapCharToString(GetConfigInfoChar());
}

void RunnerConfig::SetConfigPath(const std::string &config_path) { SetConfigPath(StringToChar(config_path)); }

std::string RunnerConfig::GetConfigPath() const { return CharToString(GetConfigPathChar()); }

class ModelParallelRunnerImpl;

/// \brief The ModelParallelRunner class is used to define a MindSpore ModelParallelRunner, facilitating Model
/// management.
class MS_API ModelParallelRunner {
 public:
  ModelParallelRunner();
  ~ModelParallelRunner();

  /// \brief build a model parallel runner from model path so that it can run on a device.
  ///
  /// \param[in] model_path Define the model path.
  /// \param[in] runner_config Define the config used to store options during model pool init.
  ///
  /// \return Status.
  inline Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  /// \brief build a model parallel runner from model buffer so that it can run on a device.
  ///
  /// \param[in] model_data Define the buffer read from a model file.
  /// \param[in] data_size Define bytes number of model buffer.
  /// \param[in] runner_config Define the config used to store options during model pool init.
  ///
  /// \return Status.
  Status Init(const void *model_data, const size_t data_size,
              const std::shared_ptr<RunnerConfig> &runner_config = nullptr);

  /// \brief Obtains all input tensors information of the model.
  ///
  /// \return The vector that includes all input tensors.
  std::vector<MSTensor> GetInputs();

  /// \brief Obtains all output tensors information of the model.
  ///
  /// \return The vector that includes all output tensors.
  std::vector<MSTensor> GetOutputs();

  /// \brief Inference ModelParallelRunner.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

 private:
  Status Init(const std::vector<char> &model_path, const std::shared_ptr<RunnerConfig> &runner_config);
  std::shared_ptr<ModelParallelRunnerImpl> model_parallel_runner_impl_ = nullptr;
};

Status ModelParallelRunner::Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config) {
  return Init(StringToChar(model_path), runner_config);
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_PARALLEL_RUNNER_H
