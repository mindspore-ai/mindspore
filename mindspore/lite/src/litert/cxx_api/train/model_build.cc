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

#include "include/api/model.h"
#include "src/common/log_adapter.h"
#include "src/litert/cxx_api/model/model_impl.h"
namespace mindspore {
Status Model::Build(GraphCell lossGraphCell, Node *optimizer, std::vector<Expr *> inputs,
                    const std::shared_ptr<Context> &model_context, const std::shared_ptr<TrainCfg> &train_cfg) {
  std::stringstream err_msg;
  if (impl_ == nullptr) {
    impl_ = std::make_shared<ModelImpl>();
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Model implement is null.";
      return kLiteFileError;
    }
  }
  auto lossGraph = lossGraphCell.GetGraph();
  if (lossGraph == nullptr) {
    err_msg << "Invalid null graph";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  impl_->SetContext(model_context);
  impl_->SetConfig(train_cfg);
  impl_->SetGraph(lossGraph);
  auto graph = impl_->BuildTrain(optimizer, inputs);
  auto status = Build(GraphCell(*graph), model_context, train_cfg);
  if (status != mindspore::kSuccess) {
    MS_LOG(ERROR) << "Error " << status << " during model build";
    return status;
  }
  return kSuccess;  // status
}

Status Model::BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context,
                                    const std::shared_ptr<TrainCfg> &train_cfg) {
  std::stringstream err_msg;
  if (impl_ == nullptr) {
    impl_ = std::make_shared<ModelImpl>();
    if (impl_ == nullptr) {
      MS_LOG(ERROR) << "Model implement is null.";
      return kLiteFileError;
    }
  }

  if (backbone.GetGraph() == nullptr || head.GetGraph() == nullptr) {
    err_msg << "Invalid null graph.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  if (context == nullptr) {
    err_msg << "Invalid null context.";
    MS_LOG(ERROR) << err_msg.str();
    return Status(kLiteNullptr, err_msg.str());
  }
  impl_->SetContext(context);
  impl_->SetGraph(head.GetGraph());
  impl_->SetConfig(train_cfg);

  Status ret = impl_->BuildTransferLearning(backbone.GetGraph(), head.GetGraph());
  if (ret != kSuccess) {
    return ret;
  }
  return kSuccess;
}
}  // namespace mindspore
