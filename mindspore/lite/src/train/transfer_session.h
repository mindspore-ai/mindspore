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
#include "src/ops/primitive_c.h"
#include "include/train_session.h"
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
  TransferSession();
  explicit TransferSession(lite::LiteSession *backend_session);
  ~TransferSession();

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  void BindThread(bool if_bind) override;
  std::vector<tensor::MSTensor *> GetInputs() const override { return lite::LiteSession::GetInputs(); }
  mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &tensor_name) const override {
    return lite::LiteSession::GetInputsByTensorName(tensor_name);
  }

 protected:
  lite::LiteSession *backend_session_;

 private:
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRANSFER_SESSION_H_
