/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/cxx_api/model/model_impl.h"
#include "include/train/train_cfg.h"
#include "src/litert/cxx_api/converters.h"
#include "src/train/transfer_session.h"
namespace mindspore {
Status ModelImpl::BuildTransferLearning(const std::shared_ptr<Graph> &backbone, const std::shared_ptr<Graph> &head) {
  const auto b_graph_data = backbone->graph_data_;
  const auto h_graph_data = head->graph_data_;
  if (b_graph_data == nullptr || h_graph_data == nullptr) {
    MS_LOG(ERROR) << "graph data cannot be nullptr";
    return kLiteNullptr;
  }
  bool is_train_session = h_graph_data->IsTrainModel();
  if (is_train_session) {
    const auto b_model = reinterpret_cast<lite::LiteModel *>(b_graph_data->lite_model().get());
    const auto h_model = reinterpret_cast<lite::LiteModel *>(h_graph_data->lite_model().get());
    if (h_model == nullptr || h_model->buf == nullptr || b_model == nullptr || b_model->buf == nullptr) {
      MS_LOG(ERROR) << "Lite model has been freed.";
      return kLiteNullptr;
    }

    lite::TrainCfg train_cfg;
    if (cfg_ != nullptr) {
      auto status = A2L_ConvertConfig(cfg_.get(), &train_cfg);
      if (status != kSuccess) {
        MS_LOG(ERROR) << "Failed to convert Config to Lite Config";
        return status;
      }
    }

    auto session = std::shared_ptr<lite::LiteSession>(
      CreateTransferSessionInt(b_model->buf, b_model->buf_size_, h_model->buf, h_model->buf_size_,
                               ContextUtils::Convert(context_.get()), true, &train_cfg));
    if (session == nullptr) {
      MS_LOG(ERROR) << "create session failed";
      return kLiteMemoryFailed;
    }
    session_.swap(session);
    return kSuccess;
  }
  MS_LOG(DEBUG) << "Session is not a train session.";
  return kLiteError;
}
}  // namespace mindspore
