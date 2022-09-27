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

#include "minddata/dataset/engine/opt/pre/add_skip_pass.h"

#include <algorithm>
#include <string>

#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/data_queue_node.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// constructor
AddSkipPass::InjectionFinder::InjectionFinder(const std::shared_ptr<DatasetNode> &node) : injection_point_(nullptr) {}

// Performs finder work for BuildVocabOp that has special rules about skip injection
Status AddSkipPass::InjectionFinder::Visit(std::shared_ptr<RootNode> node, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(modified);
  CHECK_FAIL_RETURN_UNEXPECTED(node->Children().size() > 0,
                               "Invalid data, the number of children should be greater than zero.");
  // The injection is at the child of the root node
  injection_point_ = node->Children()[0];
  num_epochs_ = node->num_epochs();
  step_ = node->step();
  return Status::OK();
}

// Performs finder work for BuildVocabOp that has special rules about skip injection
Status AddSkipPass::InjectionFinder::Visit(std::shared_ptr<BuildVocabNode> node, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(modified);
  injection_point_ = nullptr;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
// Performs finder work for BuildSentencePieceVocabNode that has special rules about skip injection
Status AddSkipPass::InjectionFinder::Visit(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(modified);
  injection_point_ = nullptr;
  return Status::OK();
}
#endif

Status AddSkipPass::InjectionFinder::VisitAfter(std::shared_ptr<DataQueueNode> node, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(modified);
  CHECK_FAIL_RETURN_UNEXPECTED(node->Children().size() > 0,
                               "Invalid data, the number of children should be greater than zero.");
  // Assumption: There is only one DataQueueNode in a pipeline. This assumption is not validated here.
  // Move the injection point to the child of this node.
  injection_point_ = node->Children()[0];
  return Status::OK();
}

// Runs an injection pass to inject in operators needed at the pre pass stage
Status AddSkipPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(root_ir);
  RETURN_UNEXPECTED_IF_NULL(modified);
  MS_LOG(INFO) << "Pre pass: AddSkipPass started.";

  // First, run the finder to perform any injection info before we can go ahead to drive the op injection work.
  // The finder can make updates to the AddSkipPass object.
  AddSkipPass::InjectionFinder finder(root_ir);
  RETURN_IF_NOT_OK(finder.Run(root_ir, modified));

  // The first injection logic is to check if we should inject the skip op as the root node.
  std::shared_ptr<DatasetNode> node = finder.injection_point();
  CHECK_FAIL_RETURN_UNEXPECTED(node != nullptr, "Failed to inject SkipOp.");

  int64_t dataset_size = -1;
  RETURN_IF_NOT_OK(root_ir->GetDatasetSize(nullptr, false, &dataset_size));
  CHECK_FAIL_RETURN_UNEXPECTED(dataset_size > 0, "Cannot reset the pipeline, dataset size is undefined");
  int32_t num_epochs = finder.GetNumEpochs();
  int64_t step = finder.GetStep();
  CHECK_FAIL_RETURN_UNEXPECTED(step >= 0,
                               "Cannot reset the pipeline, reset step must be >= 0. step: " + std::to_string(step));
  if (step >= dataset_size * num_epochs) {
    std::string err_msg = "Cannot reset the pipeline, reset step must be less than dataset_size * num_epochs. step: " +
                          std::to_string(step) + ", dataset_size: " + std::to_string(dataset_size) +
                          ", num_epochs: " + std::to_string(num_epochs);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (step == 0) {
    return Status::OK();
  }
  // in fast recovery, we start from current epoch and skip remaining steps (skip node will also be pushed down)
  if (GlobalContext::config_manager()->fast_recovery()) {
    int32_t new_num_epochs = num_epochs - static_cast<int32_t>(step / dataset_size);
    int64_t skip_num = step % dataset_size;

    root_ir->SetNumEpochs(new_num_epochs);

    auto skip_node = std::make_shared<SkipNode>(skip_num);
    skip_node->SetOnceOnly(true);
    RETURN_IF_NOT_OK(node->InsertAbove(skip_node));
  } else {  // in non-fast we only add a skip node on top of the tree (to get same augmentations)
    auto skip_node = std::make_shared<SkipNode>(step);
    skip_node->SetOnceOnly(true);
    RETURN_IF_NOT_OK(node->InsertAbove(skip_node));
  }

  MS_LOG(INFO) << "Pre pass: AddSkipPass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
