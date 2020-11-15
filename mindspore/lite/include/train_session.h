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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include "include/lite_session.h"
#include "include/train_model.h"

namespace mindspore {
namespace session {

/// \brief TrainSession Defines a class that allows training a MindSpore model
class TrainSession : public session::LiteSession {
 public:
  /// \brief Class destructor
  virtual ~TrainSession() = default;

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] context Defines the context of the session to be created
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateSession(lite::Context *context);

  /// \brief Compile MindSpore Lite train model
  ///
  /// \note CompileTrainGraph should be called before RunGraph
  ///
  /// \param[in] model Define the model to be compiled
  ///
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h
  virtual int CompileTrainGraph(lite::TrainModel *model) = 0;

  /// \brief Export the trained model into a buffer
  ///
  /// \param[in] buf The buffer to Export into. If equal to nullptr, buf will be allocated
  /// \param[in,out] len Size of the pre-allocated buffer, and returned size of the exported buffer
  ///
  /// \return pointer to the export buffer
  virtual void *ExportToBuf(char *buf, size_t *len) const = 0;

  /// \brief Save the trained model into a flatbuffer file
  ///
  /// \param[in] filename Filename to save flatbuffer to
  ///
  /// \return 0 on success or -1 in case of error
  virtual int SaveToFile(const std::string &filename) const = 0;

  /// \brief Set model to train mode
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h
  virtual int Train() = 0;

  /// \brief Check mode of model
  ///
  /// \return boolean indication if model is in train mode
  bool IsTrain() { return train_mode_ == true; }

  /// \brief Set model to eval mode
  /// \return STATUS as an error code of compiling graph, STATUS is defined in errorcode.h
  virtual int Eval() = 0;

  /// \brief Check mode of model
  ///
  /// \return boolean indication if model is in eval mode
  bool IsEval() { return train_mode_ == false; }

 protected:
  bool train_mode_ = false;
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
