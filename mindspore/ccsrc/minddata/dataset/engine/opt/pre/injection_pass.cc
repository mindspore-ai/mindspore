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
#include "minddata/dataset/engine/opt/pre/injection_pass.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"

namespace mindspore {
namespace dataset {

// constructor
InjectionPass::InjectionFinder::InjectionFinder(InjectionPass *injection_pass) : injection_pass_(injection_pass) {}

// Performs finder work for BuildVocabOp that has special rules about epoch control injection
Status InjectionPass::InjectionFinder::PreRunOnNode(std::shared_ptr<BuildVocabOp> node, bool *modified) {
  if (injection_pass_) {
    injection_pass_->epoch_ctrl_bypass_ = true;
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("Missing outer injection pass object from inside InjectionFinder!");
  }
}

// Performs finder work for BuildSentencePieceVocabOp that has special rules about epoch control injection
Status InjectionPass::InjectionFinder::PreRunOnNode(std::shared_ptr<BuildSentencePieceVocabOp> node, bool *modified) {
  if (injection_pass_) {
    injection_pass_->epoch_ctrl_bypass_ = true;
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("Missing outer injection pass object from inside InjectionFinder!");
  }
}

// Temporary code to prevent the injection of epoch control when cache op is present
// Remove this code in cache op phase 2
Status InjectionPass::InjectionFinder::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified) {
  if (injection_pass_) {
    injection_pass_->epoch_ctrl_bypass_ = true;
    return Status::OK();
  } else {
    RETURN_STATUS_UNEXPECTED("Missing outer injection pass object from inside InjectionFinder!");
  }
}

// constructor
InjectionPass::InjectionPass() : epoch_ctrl_bypass_(false) {}

// Runs an injection pass to inject in operators needed at the pre pass stage
Status InjectionPass::RunOnTree(ExecutionTree *tree, bool *modified) {
  MS_LOG(INFO) << "Pre pass: Injection pass started.";

  // First, run the finder to perform any injection info before we can go ahead to drive the op injection work.
  // The finder can make updates to the InjectionPass object.
  InjectionPass::InjectionFinder finder(this);
  finder.Run(tree, modified);

  // The first injection logic is to check if we should inject the epoch control op as the root node.
  // Do not inject the op if the number of epochs is 1.
  int32_t num_epochs = tree->num_epochs();
  if (num_epochs != 1 && !epoch_ctrl_bypass_) {
    std::shared_ptr<EpochCtrlOp> epoch_ctrl_op;
    RETURN_IF_NOT_OK(EpochCtrlOp::Builder(num_epochs).Build(&epoch_ctrl_op));
    RETURN_IF_NOT_OK(tree->AssociateNode(epoch_ctrl_op));
    std::shared_ptr<DatasetOp> node = tree->root();
    if (std::dynamic_pointer_cast<DeviceQueueOp>(node) == nullptr) {
      tree->root()->InsertAsParent(epoch_ctrl_op);
    } else {
      tree->root()->child(0)->InsertAsParent(epoch_ctrl_op);
    }
  }

  MS_LOG(INFO) << "Pre pass: Injection pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
