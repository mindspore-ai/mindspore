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

#include <vector>
#include <algorithm>
#include "minddata/dataset/engine/opt/pre/epoch_ctrl_pass.h"
#include "minddata/dataset/engine/ir/datasetops/epoch_ctrl_node.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"

namespace mindspore {
namespace dataset {

// constructor
EpochCtrlPass::InjectionFinder::InjectionFinder(std::shared_ptr<DatasetNode> node)
    : injection_point_(nullptr), num_epochs_(-1) {}

// Performs finder work for BuildVocabOp that has special rules about epoch control injection
Status EpochCtrlPass::InjectionFinder::Visit(std::shared_ptr<RootNode> node, bool *const modified) {
  // The injection is at the child of the root node
  injection_point_ = node->Children()[0];
  num_epochs_ = node->num_epochs();
  return Status::OK();
}

// Performs finder work for BuildVocabOp that has special rules about epoch control injection
Status EpochCtrlPass::InjectionFinder::Visit(std::shared_ptr<BuildVocabNode> node, bool *const modified) {
  injection_point_ = nullptr;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// Performs finder work for BuildSentencePieceVocabNode that has special rules about epoch control injection
Status EpochCtrlPass::InjectionFinder::Visit(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified) {
  injection_point_ = nullptr;
  return Status::OK();
}
#endif

Status EpochCtrlPass::InjectionFinder::VisitAfter(std::shared_ptr<TransferNode> node, bool *const modified) {
  // Assumption: There is only one TransferNode in a pipeline. This assumption is not validated here.
  // Move the injection point to the child of this node.
  injection_point_ = node->Children()[0];
  return Status::OK();
}

// constructor
EpochCtrlPass::EpochCtrlPass() {}

// Runs an injection pass to inject in operators needed at the pre pass stage
Status EpochCtrlPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  MS_LOG(INFO) << "Pre pass: Injection pass started.";

  // First, run the finder to perform any injection info before we can go ahead to drive the op injection work.
  // The finder can make updates to the EpochInjectionPass object.
  EpochCtrlPass::InjectionFinder finder(root_ir);
  RETURN_IF_NOT_OK(finder.Run(root_ir, modified));

  // The first injection logic is to check if we should inject the epoch control op as the root node.
  // Do not inject the op if the number of epochs is 1.
  std::shared_ptr<DatasetNode> node = finder.injection_point();
  int32_t num_epochs = finder.num_epochs();
  if (num_epochs != 1 && node != nullptr) {
    auto epoch_ctrl_node = std::make_shared<EpochCtrlNode>(num_epochs);
    RETURN_IF_NOT_OK(node->InsertAbove(epoch_ctrl_node));
  }
  MS_LOG(INFO) << "Pre pass: Injection pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
