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

#include "dataset/engine/opt/pass.h"
#include "dataset/engine/datasetops/batch_op.h"
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/datasetops/device_queue_op.h"
#include "dataset/engine/datasetops/map_op.h"
#include "dataset/engine/datasetops/project_op.h"
#include "dataset/engine/datasetops/rename_op.h"
#include "dataset/engine/datasetops/filter_op.h"
#include "dataset/engine/datasetops/repeat_op.h"
#include "dataset/engine/datasetops/skip_op.h"
#include "dataset/engine/datasetops/shuffle_op.h"
#include "dataset/engine/datasetops/source/generator_op.h"
#include "dataset/engine/datasetops/source/mindrecord_op.h"
#include "dataset/engine/datasetops/source/storage_op.h"
#include "dataset/engine/datasetops/source/tf_reader_op.h"
#include "dataset/engine/datasetops/source/image_folder_op.h"
#include "dataset/engine/datasetops/take_op.h"
#include "dataset/engine/datasetops/zip_op.h"

namespace mindspore {
namespace dataset {

// Driver method for TreePass
Status TreePass::Run(ExecutionTree *tree, bool *modified) {
  if (!tree || !modified) {
    return Status(StatusCode::kUnexpectedError, "Null pointer passed to TreePass");
  }
  return this->RunOnTree(tree, modified);
}

// Driver method for NodePass
Status NodePass::Run(ExecutionTree *tree, bool *modified) {
  if (!tree || !modified) {
    return Status(StatusCode::kUnexpectedError, "Null pointer passed to NodePass");
  }
  std::shared_ptr<DatasetOp> root = tree->root();
  if (traversalOrder_ == Order::DFS) {
    // DFS
    return DFSNodeVisit(root, modified);
  } else if (traversalOrder_ == Order::BFS) {
    // BFS
    return BFSNodeVisit(root, modified);
  }
  return Status::OK();
}

// Helper function to perform DFS visit
Status NodePass::DFSNodeVisit(std::shared_ptr<DatasetOp> node, bool *modified) {
  for (const auto &c : node->Children()) {
    RETURN_IF_NOT_OK(this->DFSNodeVisit(c, modified));
  }
  return node->Accept(this, modified);
}

// Helper function to perform BFS visit
Status NodePass::BFSNodeVisit(std::shared_ptr<DatasetOp> root, bool *modified) {
  // Initialize bfs queue with root
  std::queue<std::shared_ptr<DatasetOp>> bfsQueue;
  bfsQueue.push(root);

  // BFS loop
  while (!bfsQueue.empty()) {
    // Pop the front of the bfs queue
    auto curNode = bfsQueue.front();
    bfsQueue.pop();

    // Run node pass
    RETURN_IF_NOT_OK(curNode->Accept(this, modified));

    // Push children into bfs queue
    for (const auto &c : curNode->Children()) {
      bfsQueue.push(c);
    }
  }
  return Status::OK();
}

Status NodePass::RunOnNode(std::shared_ptr<BatchOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<MapOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ProjectOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<RenameOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<FilterOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<SkipOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<TFReaderOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<TakeOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ZipOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

}  // namespace dataset
}  // namespace mindspore
