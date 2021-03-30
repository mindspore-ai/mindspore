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
#ifndef MINDSPORE_LITE_SRC_TRAIN_TRANSFER_SESSION_H_
#define MINDSPORE_LITE_SRC_TRAIN_TRANSFER_SESSION_H_
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include "src/train/train_model.h"
#include "src/lite_session.h"
#include "src/train/train_session.h"

/*
                 Inheritance Diagram

            +-------------------------------+
            |     session::LiteSession      |
            +--------+------------+---------+
                    /              \
 +-----------------+-----+  +-------+------------+
 | session::TrainSession |  | lite::LiteSession  |
 +-----------------+-----+  +-------+------------+
                    \              /
            +--------+------------+---------+
            |       lite::TrainSession      |
            +-------------------------------+
                            |
            +--------+------------+---------+
            |       lite::TrasferSession    |
            +-------------------------------+
*/

namespace mindspore {
namespace lite {

class TransferSession : public lite::TrainSession {
 public:
  explicit TransferSession(const char *model_buf_backbone, size_t size_backbone, lite::Context *context);

  ~TransferSession();

  bool is_valid() const { return is_valid_; }

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  void BindThread(bool if_bind) override;
  std::vector<tensor::MSTensor *> GetInputs() const override;
  mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &tensor_name) const override;

  int CompileTransferGraph();

 protected:
  lite::LiteSession *backbone_session_ = nullptr;
  char *lite_model_ = nullptr;
  std::vector<mindspore::tensor::MSTensor *> combined_inputs_;
  std::vector<std::pair<mindspore::tensor::MSTensor *, mindspore::tensor::MSTensor *>> backbone_head_map_;
  bool is_valid_ = false;

 private:
  bool CompileFormatTransform(tensor::MSTensor *out, tensor::MSTensor *in, int *mask, size_t mask_len);
  bool nchw2nhwc_ = false;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRANSFER_SESSION_H_
