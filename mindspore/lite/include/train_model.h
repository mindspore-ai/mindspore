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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_MODEL_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_MODEL_H_
#include <vector>
#include "include/model.h"

namespace mindspore::lite {
struct TrainModel : public lite::Model {
  /// \brief Static method to create a TrainModel pointer.
  ///
  /// \param[in] model_buf Define the buffer read from a model file.
  /// \param[in] size Define bytes number of model buffer.
  ///
  /// \return Pointer of MindSpore Lite TrainModel.
  static TrainModel *Import(const char *model_buf, size_t size);

  /// \brief Free meta graph temporary buffer
  void Free() override;

  /// \brief TrainModel destruct, free all memory
  virtual ~TrainModel();

  /// \brief Export Model into buf.
  ///
  /// \param[in] buf Define the buffer to Export into. If nullptr, buf will be allocated
  /// \param[in] len size of the buffer.
  ///
  /// \return Pointer to buffer with exported model
  char* ExportBuf(char* buf, size_t* len) const;

  size_t buf_size_;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_MODEL_H_
