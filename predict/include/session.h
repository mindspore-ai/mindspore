/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_INCLUDE_SESSION_H_
#define PREDICT_INCLUDE_SESSION_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include "include/context.h"
#include "include/tensor.h"

#define MSPREDICT_API __attribute__((visibility("default")))

namespace mindspore {
namespace predict {
using NODE_ID = std::string;

///\brief Graph defined by MindSpore predict.
///
///\note
/// The caller does not need to care about detailed implementation of this class, so just list the class name here.
class Graph;

///\brief GraphExecution defined by MindSpore predict.
///
///\note
/// The caller does not need to care about detailed implementation of this class, so just list the class name here.
class GraphExecution;

///\brief MindSpore predict session.
///
/// This class represents session of MindSpore predict.
///
///\note
/// The caller needs to allocate and free memory of inputs and outputs.
/// New Session is not suggested, please use CreateSession function to create new session class.
class MSPREDICT_API Session {
 public:
  ///\brief Constructor of MindSpore predict session.
  ///
  ///\param[in] ctx The context of the session.
  ///
  ///\return Instance of MindSpore predict session.
  explicit Session(const Context &ctx);

  ///\brief Destructor of MindSpore predict session.
  ~Session();

  ///\brief Init the session.
  ///
  ///\param[in] ctx The context of the session.
  ///\param[in] size The size of the session.
  ///\param[in] graphBuf The buffer of the graph, used for build session.
  ///
  ///\return Return RET_OK if the initialization is success, otherwhise return RET_ERROR.
  int Init(const char *graphBuf, size_t size);

  ///\brief Get the input of session.
  ///
  ///\return Input node's input tensors if found, empty vector otherwise.
  ///
  ///\note
  /// The caller needs to allocate and free memory of inputs.
  std::vector<Tensor *> GetInput();

  ///\brief Run the session.
  ///
  ///\param[in] inputs The input of the session.
  ///
  ///\return Return RET_OK if run success, otherwhise return RET_ERROR.
  ///\note
  /// Currently input tensors' data format only support FORMAT_NCHW.
  /// Currently input tensors' data type only support FLOAT.
  int Run(const std::vector<Tensor *> &inputs);

  ///\brief Get the output of session.
  ///
  ///\param[in] nodeName Given output node name.
  ///
  ///\return Output node's output tensors if found, empty vector otherwise.
  ///
  ///\note
  /// The caller needs to free memory of outputs.
  std::vector<Tensor *> GetOutput(const std::string &nodeName);

  ///\brief Get the all output of session.
  ///
  ///\return Every output node's output tensors.
  ///
  ///\note
  /// The caller needs to free memory of outputs.
  std::map<std::string, std::vector<Tensor *>> GetAllOutput();

 protected:
  ///\brief Init the executor.
  ///
  ///\return Return RET_OK if the initialization is success, otherwhise return RET_ERROR.
  int InitExecutor();

  const Context &_ctx;
  Graph *_graph = nullptr;
  GraphExecution *_executor = nullptr;
  bool reinitExecutor = true;
};

///\brief MindSpore predict neural network session create function
///
/// This function used to create MindSpore predict neural network session, which will be used to run the neural network.
///
///\param[in] sessionName The name of the session.
///\param[in] graphBuf The buffer of the graph, used for build session.
///\param[in] size The size of the session.
///\param[in] ctx The context of the session.
///
///\return Instance of MindSpore predict session.
///
///\note
/// The caller needs to allocate and free memory of graph buffer.
std::shared_ptr<Session> MSPREDICT_API CreateSession(const char *graphBuf, size_t size, const Context &ctx);
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_INCLUDE_SESSION_H_
