/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <string>
#include "include/lite_session.h"

namespace mindspore {
namespace session {
class TrainSession {
 public:
  /// \brief Static method to create a TransferSession object
  ///
  /// \param[in] filename_backbone Filename to read backbone net flatbuffer from
  /// \param[in] filename_head Filename to read head net flatbuffer from
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  ///
  /// \return Pointer of MindSpore LiteSession
  static LiteSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head,
                                            const lite::Context *context, bool train_mode = false,
                                            const lite::TrainCfg *cfg = nullptr);

  /// \brief Static method to create a TrainSession object
  ///
  /// \param[in] filename name of flatbuffer that holds the flatbuffer
  /// \param[in] context Defines the context of the session to be created
  /// \param[in] train_mode training mode to initialize Session with
  /// \param[in] cfg training configuration, set to null for default configuration
  ///
  /// \return Pointer of MindSpore LiteSession
  static LiteSession *CreateTrainSession(const std::string &filename, const lite::Context *context,
                                         bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_TRAIN_SESSION_H_
