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
#include "include/lite_session.h"
#include "include/errorcode.h"

namespace mindspore {
namespace session {

/// \brief TrainSession Defines a class that allows training a MindSpore model
class TrainSession : public session::LiteSession {
 public:
  /// \brief Class destructor
  virtual ~TrainSession() = default;

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] model_buf A buffer that was read from a MS model file
  /// \param[in] size Length of the buffer
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateSession(const char *model_buf, size_t size, lite::Context *context,
                                     bool train_mode = false);

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] filename Filename to read flatbuffer from
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateSession(const std::string &filename, lite::Context *context, bool train_mode = false);

  /// \brief Static method to create a transfer lernning support TrainSession object
  ///
  /// \param[in] model_buf_backbone A buffer that was read from a backbone MS model file
  /// \param[in] size_backbone Length of the backbone net buffer
  /// \param[in] model_buf_head A buffer that was read from a head MS model file
  /// \param[in] size_head Length of the head net buffer
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateTransferSession(const char *model_buf_backbone, size_t size_backbone,
                                             const char *model_buf_head, size_t size_head, lite::Context *context,
                                             bool train_mode = false);

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] filename_backbone Filename to read backbone net flatbuffer from
  /// \param[in] filename_head Filename to read head net flatbuffer from
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head,
                                             lite::Context *context, bool train_mode = false);

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

  /// \brief Sets the Learning Rate of the training
  ///
  /// \param[in] learning_rate to set
  ///
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int SetLearningRate(float learning_rate) = 0;

  /// \brief Gets the Learning Rate of the training
  ///
  /// \return learning rate. 0.0 if no optimizer was found
  virtual float GetLearningRate() = 0;

  /// \brief Setup training with virtual batches
  ///
  /// \param[in] virtual_batch_multiplier - virtual batch multiplier, use any number < 1 to disable
  /// \param[in] lr - learning rate to use for virtual batch, -1 for internal configuration
  /// \param[in] momentum - batch norm momentum to use for virtual batch, -1 for internal configuration

  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) = 0;

  /// \brief Get output MindSpore Lite MSTensors of Training model prediction
  ///
  /// \return a vector of output tensors (MindSpore Lite MSTensor).
  virtual std::vector<tensor::MSTensor *> GetPredictions() const = 0;

  /// \brief Set part of the name that identify a loss kernel
  /// \param[in] loss_name Identifucation name for loss kernels
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int SetLossName(std::string loss_name) {
    loss_name_ = loss_name;
    return mindspore::lite::RET_OK;
  }

 protected:
  bool train_mode_ = false;
  std::string get_loss_name() const { return loss_name_; }

 private:
  std::string loss_name_ = "_loss_fn";
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_SESSION_H_
