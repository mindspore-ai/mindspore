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
#include "minddata/dataset/engine/opt/pre/epoch_injection_pass.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"

namespace mindspore {
namespace dataset {

// constructor
EpochInjectionPass::InjectionFinder::InjectionFinder(std::shared_ptr<DatasetOp> node) : injection_point_(node) {}

#ifndef ENABLE_ANDROID
// Performs finder work for BuildVocabOp that has special rules about epoch control injection
Status EpochInjectionPass::InjectionFinder::PreRunOnNode(std::shared_ptr<BuildVocabOp> node, bool *const modified) {
  injection_point_ = nullptr;
  return Status::OK();
}

// Performs finder work for BuildSentencePieceVocabOp that has special rules about epoch control injection
Status EpochInjectionPass::InjectionFinder::PreRunOnNode(std::shared_ptr<BuildSentencePieceVocabOp> node,
                                                         bool *const modified) {
  injection_point_ = nullptr;
  return Status::OK();
}
#endif

Status EpochInjectionPass::InjectionFinder::RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *const modified) {
  // Assumption: There is only one DeviceQueueOp in a pipeline. This assumption is not validated here.
  injection_point_ = node->child(0);
  return Status::OK();
}

// constructor
EpochInjectionPass::EpochInjectionPass() {}

// Runs an injection pass to inject in operators needed at the pre pass stage
Status EpochInjectionPass::RunOnTree(ExecutionTree *tree, bool *const modified) {
  MS_LOG(INFO) << "Pre pass: Injection pass started.";

  // First, run the finder to perform any injection info before we can go ahead to drive the op injection work.
  // The finder can make updates to the EpochInjectionPass object.
  EpochInjectionPass::InjectionFinder finder(tree->root());
  RETURN_IF_NOT_OK(finder.Run(tree, modified));

  // The first injection logic is to check if we should inject the epoch control op as the root node.
  // Do not inject the op if the number of epochs is 1.
  int32_t num_epochs = tree->num_epochs();
  std::shared_ptr<DatasetOp> epoch_inject_node = finder.injection_point();
  if (num_epochs != 1 && epoch_inject_node != nullptr) {
    std::shared_ptr<EpochCtrlOp> epoch_ctrl_op;
    RETURN_IF_NOT_OK(EpochCtrlOp::Builder(num_epochs).Build(&epoch_ctrl_op));
    RETURN_IF_NOT_OK(tree->AssociateNode(epoch_ctrl_op));
    RETURN_IF_NOT_OK(epoch_inject_node->InsertAsParent(epoch_ctrl_op));
  }

  MS_LOG(INFO) << "Pre pass: Injection pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
