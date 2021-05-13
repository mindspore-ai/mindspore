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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_SESSION_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_SESSION_H_
#include <vector>
#include <string>
#include <tuple>
#include "include/lite_session.h"
#include "include/errorcode.h"
#include "include/train/train_cfg.h"

namespace mindspore {
namespace session {

/// \brief TrainSession Defines a class that allows training a MindSpore model
class TrainSession : public session::LiteSession {
 public:
  /// \brief Class destructor
  virtual ~TrainSession() = default;

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] filename name of flatbuffer that holds the flatbuffer
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  /// \param[in] cfg training configuration, set to null for default configuration
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateSession(const std::string &filename, const lite::Context *context, bool train_mode = false,
                                     const lite::TrainCfg *cfg = nullptr);

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] filename_backbone Filename to read backbone net flatbuffer from
  /// \param[in] filename_head Filename to read head net flatbuffer from
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  ///
  /// \return Pointer of MindSpore Lite TrainSession
  static TrainSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head,
                                             const lite::Context *context, bool train_mode = false,
                                             const lite::TrainCfg *cfg = nullptr);

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

  /// \brief Save model
  /// \param[in] file_name pretrained model file name prefix. '.ms' extenension is added if does not exist
  /// \param[in] model_type indication whether to save full model or only the inference part
  /// \param[in] quant_type indication whether to quantize exported model
  /// \param[in] format of exported file (currently only FT_FLATBUFFER is supported)
  /// \return STATUS as an error code of the set operation, STATUS is defined in errorcode.h
  virtual int Export(const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantType quant_type = lite::QT_DEFAULT, lite::FormatType = lite::FT_FLATBUFFER) {
    return mindspore::lite::RET_ERROR;
  }

 protected:
  bool train_mode_ = false;
};

}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_SESSION_H_
