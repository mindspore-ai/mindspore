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
#ifndef MINDSPORE_INCLUDE_API_MODEL_H
#define MINDSPORE_INCLUDE_API_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>
#include "include/api/status.h"
#include "include/api/types.h"
#include "include/api/graph.h"
#include "include/api/context.h"
#include "include/api/callback/callback.h"
#include "include/api/cell.h"
#include "include/api/cfg.h"
#include "include/api/dual_abi_helper.h"

namespace mindspore {
class ModelImpl;
class Metrics;
class Net;
class Node;
class Expr;

namespace dataset {
class Dataset;
}  // namespace dataset
/// \brief The Model class is used to define a MindSpore model, facilitating computational graph management.
class MS_API Model {
 public:
  Model();
  ~Model();
  Model(const Model &) = delete;
  void operator=(const Model &) = delete;

  /// \brief Build a model from model buffer so that it can run on a device.
  ///
  /// \param[in] model_data Define the buffer read from a model file.
  /// \param[in] data_size Define bytes number of model buffer.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  ///
  /// \return Status.
  Status Build(const void *model_data, size_t data_size, ModelType model_type,
               const std::shared_ptr<Context> &model_context = nullptr);

  /// \brief Load and build a model from model buffer so that it can run on a device.
  ///
  /// \param[in] model_path Define the model path.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  ///
  /// \return Status.
  inline Status Build(const std::string &model_path, ModelType model_type,
                      const std::shared_ptr<Context> &model_context = nullptr);

  /// \brief Build a model from model buffer so that it can run on a device.
  ///
  /// \param[in] model_data Define the buffer read from a model file.
  /// \param[in] data_size Define bytes number of model buffer.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  /// \param[in] dec_key Define the key used to decrypt the ciphertext model. The key length is 16.
  /// \param[in] dec_mode Define the decryption mode. Options: AES-GCM.
  /// \param[in] cropto_lib_path Define the openssl library path.
  ///
  /// \return Status.
  inline Status Build(const void *model_data, size_t data_size, ModelType model_type,
                      const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::string &dec_mode,
                      const std::string &cropto_lib_path);

  /// \brief Load and build a model from model buffer so that it can run on a device.
  ///
  /// \param[in] model_path Define the model path.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  /// \param[in] dec_key Define the key used to decrypt the ciphertext model. The key length is 16.
  /// \param[in] dec_mode Define the decryption mode. Options: AES-GCM.
  /// \param[in] cropto_lib_path Define the openssl library path.
  ///
  /// \return Status.
  inline Status Build(const std::string &model_path, ModelType model_type,
                      const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::string &dec_mode,
                      const std::string &cropto_lib_path);

  /// \brief Build a model
  ///
  /// \param[in] graph GraphCell is a derivative of Cell. Cell is not available currently. GraphCell can be constructed
  /// from Graph, for example, model.Build(GraphCell(graph), context).
  /// \param[in] model_context A context used to store options during execution.
  /// \param[in] train_cfg A config used by training.
  ///
  /// \return Status.
  Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context = nullptr,
               const std::shared_ptr<TrainCfg> &train_cfg = nullptr);

  /// \brief Build train model
  ///
  /// \param[in] graph A forward network
  /// \param[in] optimizer An optimizer node
  /// \param[in] inputs Inputs expression for the trained network (ex: input, label )
  /// \param[in] model_context A context used to store options during execution.
  /// \param[in] train_cfg A config used by training
  /// \return Status

  Status Build(GraphCell graph, Node *optimizer, std::vector<Expr *> inputs,
               const std::shared_ptr<Context> &model_context, const std::shared_ptr<TrainCfg> &train_cfg);
  /// \brief Build a Transfer Learning model where the backbone weights are fixed and the head weights are trainable
  ///
  /// \param[in] backbone The static, non-learnable part of the graph
  /// \param[in] head The trainable part of the graph
  /// \param[in] context A context used to store options during execution
  /// \param[in] train_cfg A config used by training
  ///
  /// \return Status
  Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context,
                               const std::shared_ptr<TrainCfg> &train_cfg = nullptr);

  /// \brief Resize the shapes of inputs.
  ///
  /// \param[in] inputs A vector that includes all input tensors in order.
  /// \param[in] dims Defines the new shapes of inputs, should be consistent with inputs.
  ///
  /// \return Status.
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);

  /// \brief Change the size and or content of weight tensors
  ///
  /// \param[in] new_weights a vector of tensors with new shapes and data to use in the model
  ///            If data pointer is null, the data of the original tensors will be copied to the new ones
  ///
  /// \return Status.
  Status UpdateWeights(const std::vector<MSTensor> &new_weights);

  /// \brief Inference model API. If use this API in train mode, it's equal to RunStep API.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  /// \brief Inference model API. If use this API in train mode, it's equal to RunStep API.
  ///
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  Status Predict(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  /// \brief Training API. Run model by step.
  ///
  /// \param[in] before CallBack before RunStep.
  /// \param[in] after CallBack after RunStep.
  ///
  /// \return Status.
  Status RunStep(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  /// \brief Inference model with preprocess in model.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  Status PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                               const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

  /// \brief Apply data preprocess if it exits in model.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  ///
  /// \return Status.
  Status Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs);

  /// \brief Check if data preprocess exists in model.
  /// \return true if data preprocess exists.
  bool HasPreprocess();

  /// \brief Load config file.
  ///
  /// \param[in] config_path config file path.
  ///
  /// \return Status.
  inline Status LoadConfig(const std::string &config_path);

  /// \brief Update config.
  ///
  /// \param[in] section define the config section.
  /// \param[in] config define the config will be updated.
  ///
  /// \return Status.
  inline Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config);

  /// \brief Obtains all input tensors of the model.
  ///
  /// \return The vector that includes all input tensors.
  std::vector<MSTensor> GetInputs();

  /// \brief Obtains the input tensor of the model by name.
  ///
  /// \return The input tensor with the given name, if the name is not found, an invalid tensor is returned.
  inline MSTensor GetInputByTensorName(const std::string &tensor_name);

  /// \brief Obtain all gradient tensors of the model.
  ///
  /// \return The vector that includes all gradient tensors.
  std::vector<MSTensor> GetGradients() const;

  /// \brief Update gradient tensors of the model.
  ///
  /// \param[in] gradients A vector new gradients.
  ///
  /// \return Status of operation
  Status ApplyGradients(const std::vector<MSTensor> &gradients);

  /// \brief Obtain all weights tensors of the model.
  ///
  /// \return The vector that includes all weights tensors.
  std::vector<MSTensor> GetFeatureMaps() const;

  /// \brief Obtain all trainable parameters of the model optimizers.
  ///
  /// \return The vector that includes all trainable parameters.
  std::vector<MSTensor> GetTrainableParams() const;

  /// \brief Update weights tensors of the model.
  ///
  /// \param[in] new_weights A vector new weights.
  ///
  /// \return Status of operation
  Status UpdateFeatureMaps(const std::vector<MSTensor> &new_weights);

  /// \brief Obtain optimizer params tensors of the model.
  ///
  /// \return The vector that includes all params tensors.
  std::vector<MSTensor> GetOptimizerParams() const;

  /// \brief Update the optimizer parameters.
  ///
  /// \param[in] params A vector new optimizer params.
  ///
  /// \return Status of operation.
  Status SetOptimizerParams(const std::vector<MSTensor> &params);

  /// \brief Setup training with virtual batches.
  ///
  /// \param[in] virtual_batch_multiplier - virtual batch multiplier, use any number < 1 to disable.
  /// \param[in] lr - learning rate to use for virtual batch, -1 for internal configuration.
  /// \param[in] momentum - batch norm momentum to use for virtual batch, -1 for internal configuration.
  ///
  /// \return Status of operation.
  Status SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f);

  /// \brief Set the Learning Rate of the training.
  ///
  /// \param[in] learning_rate to set.
  ///
  /// \return Status of operation.
  Status SetLearningRate(float learning_rate);

  /// \brief Get the Learning Rate of the optimizer.
  ///
  /// \return Learning rate. 0.0 if no optimizer was found.
  float GetLearningRate();

  /// \brief Initialize object with metrics.
  ///
  /// \param[in] metrics A verctor of metrics objects.
  ///
  /// \return 0 on success or -1 in case of error
  Status InitMetrics(std::vector<Metrics *> metrics);

  /// \brief Accessor to TrainLoop metric objects
  ///
  /// \return A vector of metrics
  std::vector<Metrics *> GetMetrics();

  /// \brief Obtains all output tensors of the model.
  ///
  /// \return The vector that includes all output tensors.
  std::vector<MSTensor> GetOutputs();

  /// \brief Obtains names of all output tensors of the model.
  ///
  /// \return A vector that includes names of all output tensors.
  inline std::vector<std::string> GetOutputTensorNames();

  /// \brief Obtains the output tensor of the model by name.
  ///
  /// \return The output tensor with the given name, if the name is not found, an invalid tensor is returned.
  inline MSTensor GetOutputByTensorName(const std::string &tensor_name);

  /// \brief Get output MSTensors of model by node name.
  ///
  /// \param[in] node_name Define node name.
  ///
  /// \note Deprecated, replace with GetOutputByTensorName
  ///
  /// \return The vector of output MSTensor.
  inline std::vector<MSTensor> GetOutputsByNodeName(const std::string &node_name);

  /// \brief Bind GLTexture2D object to cl Memory.
  ///
  /// \param[in] inputGLTexture The input GLTexture id for Model.
  /// \param[in] outputGLTexture The output GLTexture id for Model.
  ///
  /// \return Status of operation.

  Status BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                               std::map<std::string, unsigned int> *outputGLTexture);

  /// \brief Inference model.
  ///
  /// \param[in] device_type Device type，options are kGPU, kAscend etc.
  /// \param[in] model_type The type of model file, options are ModelType::kMindIR, ModelType::kOM.
  ///
  /// \return Is supported or not.
  static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type);

  /// \brief Set the model running mode.
  ///
  /// \param[in] train True means model runs in Train Mode, otherwise Eval Mode.
  ///
  /// \return Status of operation.
  Status SetTrainMode(bool train);

  /// \brief Get the model running mode.
  ///
  /// \return Is Train Mode or not.
  bool GetTrainMode() const;

  /// \brief Performs the training Loop in Train Mode.
  ///
  /// \param[in] epochs The number of epoch to run.
  /// \param[in] ds A smart pointer to MindData Dataset object.
  /// \param[in] cbs A vector of TrainLoopCallBack objects.
  ///
  /// \return Status of operation.
  Status Train(int epochs, std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> cbs);

  /// \brief Performs the training loop over all data in Eval Mode.
  ///
  /// \param[in] ds A smart pointer to MindData Dataset object.
  /// \param[in] cbs A vector of TrainLoopCallBack objects.
  ///
  /// \return Status of operation.
  Status Evaluate(std::shared_ptr<dataset::Dataset> ds, std::vector<TrainCallBack *> cbs);

 private:
  friend class Serialization;
  // api without std::string
  MSTensor GetInputByTensorName(const std::vector<char> &tensor_name);
  std::vector<std::vector<char>> GetOutputTensorNamesChar();
  MSTensor GetOutputByTensorName(const std::vector<char> &tensor_name);
  std::vector<MSTensor> GetOutputsByNodeName(const std::vector<char> &node_name);
  Status LoadConfig(const std::vector<char> &config_path);
  Status UpdateConfig(const std::vector<char> &section, const std::pair<std::vector<char>, std::vector<char>> &config);
  Status Build(const std::vector<char> &model_path, ModelType model_type,
               const std::shared_ptr<Context> &model_context);
  Status Build(const void *model_data, size_t data_size, ModelType model_type,
               const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::vector<char> &dec_mode,
               const std::vector<char> &cropto_lib_path);
  Status Build(const std::vector<char> &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context,
               const Key &dec_key, const std::vector<char> &dec_mode, const std::vector<char> &cropto_lib_path);
  std::shared_ptr<ModelImpl> impl_;
};

MSTensor Model::GetInputByTensorName(const std::string &tensor_name) {
  return GetInputByTensorName(StringToChar(tensor_name));
}

std::vector<std::string> Model::GetOutputTensorNames() { return VectorCharToString(GetOutputTensorNamesChar()); }

MSTensor Model::GetOutputByTensorName(const std::string &tensor_name) {
  return GetOutputByTensorName(StringToChar(tensor_name));
}

std::vector<MSTensor> Model::GetOutputsByNodeName(const std::string &node_name) {
  return GetOutputsByNodeName(StringToChar(node_name));
}

Status Model::LoadConfig(const std::string &config_path) { return LoadConfig(StringToChar(config_path)); }

Status Model::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  std::pair<std::vector<char>, std::vector<char>> config_pair = {StringToChar(config.first),
                                                                 StringToChar(config.second)};
  return UpdateConfig(StringToChar(section), config_pair);
}

Status Model::Build(const void *model_data, size_t data_size, ModelType model_type,
                    const std::shared_ptr<Context> &model_context, const Key &dec_key, const std::string &dec_mode,
                    const std::string &cropto_lib_path) {
  return Build(model_data, data_size, model_type, model_context, dec_key, StringToChar(dec_mode),
               StringToChar(cropto_lib_path));
}

Status Model::Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context,
                    const Key &dec_key, const std::string &dec_mode, const std::string &cropto_lib_path) {
  auto model_path_char = StringToChar(model_path);
  return Build(model_path_char, model_type, model_context, dec_key, StringToChar(dec_mode),
               StringToChar(cropto_lib_path));
}

Status Model::Build(const std::string &model_path, ModelType model_type,
                    const std::shared_ptr<Context> &model_context) {
  return Build(StringToChar(model_path), model_type, model_context);
}
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_H
