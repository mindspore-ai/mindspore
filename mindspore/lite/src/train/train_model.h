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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRAIN_MODEL_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRAIN_MODEL_H_
#include <vector>
#include "src/lite_model.h"

namespace mindspore {
namespace lite {
/// \brief TrainModel Defines a class that allows to import and export a mindsport trainable model
struct TrainModel : public lite::LiteModel {
  /// \brief Static method to create a TrainModel object
  ///
  /// \param[in] model_buf A buffer that was read from a MS model file
  /// \param[in] size Length of the buffer
  //
  /// \return Pointer to MindSpore Lite TrainModel
  static TrainModel *Import(const char *model_buf, size_t size);

  /// \brief Free meta graph related data
  void Free() override;

  /// \brief Class destructor, free all memory
  virtual ~TrainModel() = default;

  /// \brief Export Model into a buffer
  ///
  /// \param[in] buf The buffer to Export into. If equal to nullptr, buf will be allocated
  /// \param[in,out] len Size of the pre-allocated buffer, and returned size of the exported buffer
  ///
  /// \return Pointer to buffer with exported model
  char *ExportBuf(char *buf, size_t *len) const;

  /// \brief Get Model buffer
  ///
  /// \param[in,out] len Return size of the buffer
  ///
  /// \return Pointer to model buffer
  char *GetBuffer(size_t *len) const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_TRAIN_TRAIN_MODEL_H_
