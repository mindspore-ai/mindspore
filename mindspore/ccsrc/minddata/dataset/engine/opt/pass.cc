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

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/build_sentence_piece_vocab_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/build_vocab_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/epoch_ctrl_node.h"
#include "minddata/dataset/engine/ir/datasetops/filter_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#endif
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/ir/datasetops/sync_wait_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

//////////////////////////////////
// This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
#include "minddata/dataset/engine/datasetops/batch_op.h"
#include "minddata/dataset/engine/datasetops/build_vocab_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"
#include "minddata/dataset/engine/datasetops/cache_op.h"
#include "minddata/dataset/engine/datasetops/cache_merge_op.h"
#include "minddata/dataset/engine/datasetops/cache_lookup_op.h"
#endif
#include "minddata/dataset/engine/datasetops/concat_op.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/datasetops/project_op.h"
#include "minddata/dataset/engine/datasetops/rename_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/source/album_op.h"
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#endif
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/datasetops/source/clue_op.h"
#include "minddata/dataset/engine/datasetops/source/csv_op.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#endif
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/datasetops/filter_op.h"
#include "minddata/dataset/engine/datasetops/source/generator_op.h"
#endif
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"
//////////////////////////////////

namespace mindspore {
namespace dataset {

// Driver method for TreePass
Status IRTreePass::Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  if (root_ir == nullptr || modified == nullptr) {
    return Status(StatusCode::kUnexpectedError, "Null pointer passed to TreePass");
  }
  // Initialize modified flag
  *modified = false;
  return this->RunOnTree(root_ir, modified);
}

// Driver method for NodePass
Status IRNodePass::Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  if (root_ir == nullptr || modified == nullptr) {
    return Status(StatusCode::kUnexpectedError, "Null pointer passed to NodePass");
  }
  // Initialize modified flag
  *modified = false;
  if (traversalOrder_ == Order::DFS) {
    // DFS
    return DFSNodeVisit(root_ir, modified);
  } else if (traversalOrder_ == Order::BFS) {
    // BFS
    return BFSNodeVisit(root_ir, modified);
  }
  return Status::OK();
}

// Helper function to perform DFS visit
Status IRNodePass::DFSNodeVisit(std::shared_ptr<DatasetNode> node_ir, bool *const modified) {
  bool m = false;

  RETURN_IF_NOT_OK(node_ir->Accept(this, &m));
  *modified = *modified || m;
  for (const auto &c : node_ir->Children()) {
    RETURN_IF_NOT_OK(this->DFSNodeVisit(c, &m));
    *modified = *modified || m;
  }
  RETURN_IF_NOT_OK(node_ir->AcceptAfter(this, &m));
  *modified = *modified || m;
  return Status::OK();
}

// Helper function to perform BFS visit
Status IRNodePass::BFSNodeVisit(std::shared_ptr<DatasetNode> node_ir, bool *const modified) {
  bool m = false;

  // Initialize bfs queue with root
  std::queue<std::shared_ptr<DatasetNode>> bfsQueue;
  bfsQueue.push(node_ir);

  // BFS loop
  while (!bfsQueue.empty()) {
    // Pop the front of the bfs queue
    auto curNode = bfsQueue.front();
    bfsQueue.pop();

    // Run node pass
    RETURN_IF_NOT_OK(curNode->Accept(this, &m));
    *modified = *modified || m;

    // Push children into bfs queue
    for (const auto &c : curNode->Children()) {
      bfsQueue.push(c);
    }
  }
  return Status::OK();
}

// For non-leaf IR node
Status IRNodePass::Visit(std::shared_ptr<BatchNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<BatchNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<BucketBatchByLengthNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<BucketBatchByLengthNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<BuildVocabNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<BuildVocabNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<ConcatNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<ConcatNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<FilterNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<FilterNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
#ifdef ENABLE_PYTHON
Status IRNodePass::Visit(std::shared_ptr<GeneratorNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<MappableSourceNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<GeneratorNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<MappableSourceNode>(node), modified);
}
#endif
Status IRNodePass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<MapNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
#ifndef ENABLE_ANDROID
Status IRNodePass::Visit(std::shared_ptr<MindDataNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<MappableSourceNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<MindDataNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<MappableSourceNode>(node), modified);
}
#endif
Status IRNodePass::Visit(std::shared_ptr<ProjectNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<ProjectNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<RandomNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<NonMappableSourceNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<RandomNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<NonMappableSourceNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<RenameNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<RenameNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<RepeatNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<RootNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<RootNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<ShuffleNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<ShuffleNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<SkipNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<SkipNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<TakeNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<TakeNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<TransferNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<TransferNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<ZipNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<ZipNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
#ifdef ENABLE_PYTHON
Status IRNodePass::Visit(std::shared_ptr<SyncWaitNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<SyncWaitNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
#endif
#ifndef ENABLE_ANDROID
Status IRNodePass::Visit(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
#endif

// leaf-IR Node
Status IRNodePass::Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}

Status IRNodePass::Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}

//////////////////////////////////
// This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
// Driver method for TreePass
Status TreePass::Run(ExecutionTree *tree, bool *const modified) {
  if (tree == nullptr || modified == nullptr) {
    return Status(StatusCode::kUnexpectedError, "Null pointer passed to TreePass");
  }
  // Initialize modified flag
  *modified = false;
  return this->RunOnTree(tree, modified);
}

// Driver method for NodePass
Status NodePass::Run(ExecutionTree *tree, bool *const modified) {
  if (tree == nullptr || modified == nullptr) {
    return Status(StatusCode::kUnexpectedError, "Null pointer passed to NodePass");
  }
  // Initialize modified flag
  *modified = false;
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
Status NodePass::DFSNodeVisit(std::shared_ptr<DatasetOp> node, bool *const modified) {
  RETURN_IF_NOT_OK(node->PreAccept(this, modified));
  for (const auto &c : node->Children()) {
    RETURN_IF_NOT_OK(this->DFSNodeVisit(c, modified));
  }
  return node->Accept(this, modified);
}

// Helper function to perform BFS visit
Status NodePass::BFSNodeVisit(std::shared_ptr<DatasetOp> root, bool *const modified) {
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

Status NodePass::RunOnNode(std::shared_ptr<BatchOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<MapOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ProjectOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<RenameOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<SkipOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ShuffleOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<RandomDataOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<TakeOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ZipOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<AlbumOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<MnistOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CifarOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CelebAOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CocoOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<RepeatOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<RepeatOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<BuildVocabOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<ZipOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<MapOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<ConcatOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<TakeOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<SkipOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<BatchOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

#ifndef ENABLE_ANDROID
Status NodePass::RunOnNode(std::shared_ptr<MindRecordOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<TFReaderOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CacheOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CacheMergeOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CacheLookupOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ClueOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<CsvOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<TextFileOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<CacheMergeOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<BuildSentencePieceVocabOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}
#endif

#ifdef ENABLE_PYTHON
Status NodePass::RunOnNode(std::shared_ptr<FilterOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<GeneratorOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<ManifestOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::RunOnNode(std::shared_ptr<VOCOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return RunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}

Status NodePass::PreRunOnNode(std::shared_ptr<FilterOp> node, bool *const modified) {
  // Fallback to base class visitor by default
  return PreRunOnNode(std::static_pointer_cast<DatasetOp>(node), modified);
}
#endif
//////////////////////////////////
}  // namespace dataset
}  // namespace mindspore
