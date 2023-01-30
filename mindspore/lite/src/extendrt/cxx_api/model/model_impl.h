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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_MODEL_IMPL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_MODEL_IMPL_H_
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/graph.h"
#include "include/api/serialization.h"
#include "extendrt/cxx_api/graph/graph_data.h"
#include "include/common/utils/utils.h"
#include "ir/func_graph.h"
#include "extendrt/infer_session.h"
#include "src/common/config_infos.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif
namespace mindspore {
class ConverterPlugin {
 public:
  typedef int (*ConverterFunc)(const mindspore::api::FuncGraphPtr &, const std::shared_ptr<Context> &,
                               const ConfigInfos &);
  static ConverterFunc GetConverterFunc();

 private:
  void *handle_ = nullptr;
  ConverterFunc converter_func_ = nullptr;
  static std::mutex mutex_;

  ConverterPlugin();
  ~ConverterPlugin();
  ConverterFunc GetConverterFuncInner();
};

class ModelImpl {
 public:
  ModelImpl() : graph_(nullptr), session_(nullptr), context_(nullptr) {}
  ~ModelImpl();

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
               const std::shared_ptr<Context> &model_context);

  /// \brief Build a model from model file path so that it can run on a device.
  ///
  /// \param[in] model_path Define the path of a model file.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  ///
  /// \return Status.
  Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context);

  /// \brief Resize model inputs shape and memory from specified dims.
  ///
  /// \param[in] inputs Define dst inputs tensors.
  /// \param[in] dims Define dst resize shapes.
  ///
  /// \return Status.
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);

  /// \brief Check if the model has pre-process,
  ///
  /// \return true if has pre-process, else return false.
  bool HasPreprocess();

  /// \brief Pre-process the model inputs and outputs.
  ///
  /// \param[in] inputs Define dst inputs tensors for pre-process.
  /// \param[in] outputs Define dst outputs tensors for pre-process.
  ///
  /// \return Status.
  Status Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs);

  /// \brief Inference model.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  /// \param[in] before CallBack before predict.
  /// \param[in] after CallBack after predict.
  ///
  /// \return Status.
  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before,
                 const MSKernelCallBack &after);

  /// \brief Inference model. (Deprecated, only support 4 args in future)
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  ///
  /// \return Status.
  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs);

  /// \brief Inference model. The inputs and outputs tensor must get from GetInputs and GetOutputs
  ///
  /// \return Status.
  Status Predict();

  /// \brief Inference model witsh pre-process.
  ///
  /// \param[in] inputs A vector where model inputs are arranged in sequence.
  /// \param[out] outputs Which is a pointer to a vector. The model outputs are filled in the container in sequence.
  ///
  /// \return Status.
  Status PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs);

  /// \brief Obtains all input tensors of the model.
  ///
  /// \return The vector that includes all input tensors.
  std::vector<MSTensor> GetInputs();

  /// \brief Obtains all output tensors of the model.
  ///
  /// \return The vector that includes all output tensors.
  std::vector<MSTensor> GetOutputs();

  /// \brief Obtains the input tensor of the model by name.
  ///
  /// \return The input tensor with the given name, if the name is not found, an invalid tensor is returned.
  MSTensor GetInputByTensorName(const std::string &name);

  /// \brief Obtains names of all output tensors of the model.
  ///
  /// \return A vector that includes names of all output tensors.
  std::vector<std::string> GetOutputTensorNames();

  /// \brief Obtains the output tensor of the model by name.
  ///
  /// \return The output tensor with the given name, if the name is not found, an invalid tensor is returned.
  MSTensor GetOutputByTensorName(const std::string &name);

  /// \brief Load config file.
  ///
  /// \param[in] config_path config file path.
  ///
  /// \return Status.
  Status LoadConfig(const std::string &config_path);

  /// \brief Update config.
  ///
  /// \param[in] section define the config section.
  /// \param[in] config define the config will be updated.
  ///
  /// \return Status.
  Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config);

  /// \brief Get config value.
  ///
  /// \param[in] section define the config section.
  /// \param[in] key define the config key.
  ///
  /// \return value of config as string type.
  std::string GetConfig(const std::string &section, const std::string &key);

  static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type);

 private:
  /// \brief Model build by buffer implementation, unified model build flow.
  ///
  /// \param[in] model_data Define the buffer read from a model file.
  /// \param[in] data_size Define bytes number of model buffer.
  /// \param[in] model_type Define The type of model file. Options: ModelType::kMindIR, ModelType::kOM. Only
  /// ModelType::kMindIR is valid for Lite.
  /// \param[in] model_context Define the context used to store options during execution.
  /// \param[in] model_path Define the model_path, this param is used for net and weight divided case.
  ///
  /// \return value of config as string type.
  Status BuildByBufferImpl(const void *model_data, size_t data_size, ModelType model_type,
                           const std::shared_ptr<Context> &model_context, const std::string &model_path = "");

  /// \brief Compare and optimize model online.
  ///
  /// \param[in] func_graph load from a model file.
  /// \param[in] model_context Define the context used to store options during execution.
  ///
  /// \return value of config as string type.
  Status ConvertGraphOnline(const FuncGraphPtr &func_graph, const std::shared_ptr<Context> &model_context);

  /// \brief Set Mindspore Context.
  /// This is used for load mindir file for model, turn off the infer shape flow
  ///
  void SetMsContext();

  friend class Model;
  friend class Serialization;

  // graph pointer for model
  std::shared_ptr<Graph> graph_ = nullptr;
  // infer session for model inference
  std::shared_ptr<InferSession> session_ = nullptr;
  // model context
  std::shared_ptr<Context> context_ = nullptr;
  // config info not in context
  ConfigInfos config_info_;
  std::map<std::string, TypeId> execution_plan_;
  std::mutex mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_MODEL_IMPL_H_
