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

#ifndef MINDSPORE_LITE_INCLUDE_LITE_SESSION_H
#define MINDSPORE_LITE_INCLUDE_LITE_SESSION_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "include/ms_tensor.h"
#include "include/model.h"
#include "include/context.h"

namespace mindspore {
namespace session {
/// \brief CallBackParam defined input arguments for callBack function.
struct CallBackParam {
  std::string name_callback_param; /**< node name argument */
  std::string type_callback_param; /**< node type argument */
};

/// \brief KernelCallBack defined the function pointer for callBack.
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs,
                                          std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>;

/// \brief LiteSession defined session in MindSpore Lite for compiling Model and forwarding model.
class MS_API LiteSession {
 public:
  /// \brief Static method to create a LiteSession pointer.
  ///
  /// \param[in] context Define the context of session to be created.
  ///
  /// \return Pointer of MindSpore Lite LiteSession.
  static LiteSession *CreateSession(lite::Context *context);

  /// \brief Destructor of MindSpore Lite LiteSession.
  virtual ~LiteSession() = default;

  /// \brief Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.
  ///
  /// \param[in] if_bind Define whether to bind or unbind threads.
  virtual void BindThread(bool if_bind) = 0;

  /// \brief Compile MindSpore Lite model.
  ///
  /// \note CompileGraph should be called before RunGraph.
  ///
  /// \param[in] model Define the model to be compiled.
  ///
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h.
  virtual int CompileGraph(lite::Model *model) = 0;

  /// \brief Get input MindSpore Lite MSTensors of model.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  virtual std::vector<tensor::MSTensor *> GetInputs() const = 0;

  /// \brief Get input MindSpore Lite MSTensors of model by node name.
  ///
  /// \param[in] node_name Define node name.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  virtual std::vector<tensor::MSTensor *> GetInputsByName(const std::string &node_name) const = 0;

  /// \brief Run session with callback.
  ///
  /// \param[in] before Define a call_back_function to be called before running each node.
  /// \param[in] after Define a call_back_function called after running each node.
  ///
  /// \note RunGraph should be called after CompileGraph.
  ///
  /// \return STATUS as an error code of running graph, STATUS is defined in errorcode.h.
  virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) = 0;

  /// \brief Get output MindSpore Lite MSTensors of model.
  ///
  /// \return The map of output node name and MindSpore Lite MSTensor.
  virtual std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> GetOutputs() const = 0;

  /// \brief Get output MindSpore Lite MSTensors of model by node name.
  ///
  /// \param[in] node_name Define node name.
  ///
  /// \return The vector of MindSpore Lite MSTensor.
  virtual std::vector<tensor::MSTensor *> GetOutputsByName(const std::string &node_name) const = 0;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_LITE_SESSION_H
