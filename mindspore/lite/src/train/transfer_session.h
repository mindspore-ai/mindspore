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
#include <memory>
#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include "src/litert/lite_session.h"
#include "src/train/train_session.h"

/*
       Inheritance Diagram

  +-------------------------------+
  |      session::LiteSession     |
  +--------------↑----------------+
                 |
  +--------------+----------------+
  |       lite::LiteSession       |
  +--------------↑----------------+
                 |
  +--------------+----------------+
  |       lite::TrainSession      |
  +--------------↑----------------+
                 |
  +--------------+----------------+
  |      lite::TrasferSession     |
  +-------------------------------+
*/

namespace mindspore {
namespace lite {

class TransferSession : public lite::TrainSession {
 public:
  explicit TransferSession(const char *model_buf_backbone, size_t size_backbone,
                           const std::shared_ptr<lite::InnerContext> &context);

  ~TransferSession();

  bool is_valid() const { return is_valid_; }

  int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) override;

  void BindThread(bool if_bind) override;
  std::vector<lite::Tensor *> GetInputs() const override;
  mindspore::lite::Tensor *GetInputsByTensorName(const std::string &tensor_name) const override;

  int CompileTransferGraph();
  int Export(const std::string &fb_name, ModelType model_type, QuantizationType quant_type, FormatType,
             std::vector<std::string> out_put_tensor_name = {}) override;

 protected:
  LiteSession *backbone_session_ = nullptr;
  char *lite_model_ = nullptr;
  std::vector<mindspore::lite::Tensor *> combined_inputs_;
  std::vector<std::pair<mindspore::lite::Tensor *, mindspore::lite::Tensor *>> backbone_head_map_;
  bool is_valid_ = false;

 private:
  bool CompileFormatTransform(lite::Tensor *out, lite::Tensor *in, int *mask, size_t mask_len);
  std::unordered_map<size_t, size_t> ConnectionMap();
  bool nchw2nhwc_ = false;
  size_t size_backbone_;
};

lite::LiteSession *CreateTransferSessionInt(const char *model_buf_backbone, size_t size_backbone,
                                            const char *model_buf_head, size_t size_head,
                                            const std::shared_ptr<InnerContext> &context, bool train_mode,
                                            const lite::TrainCfg *cfg);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_TRAIN_TRANSFER_SESSION_H_
