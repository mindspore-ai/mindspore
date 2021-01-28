/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/cache_node.h"
#include "minddata/dataset/engine/ir/datasetops/cache_merge_node.h"
#include "minddata/dataset/engine/ir/datasetops/cache_lookup_node.h"
#endif
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
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/ir/datasetops/sync_wait_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

namespace mindspore {
namespace dataset {

// Driver method for TreePass
Status IRTreePass::Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  if (root_ir == nullptr || modified == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, "Null pointer passed to TreePass");
  }
  // Initialize modified flag
  *modified = false;
  return this->RunOnTree(root_ir, modified);
}

// Driver method for NodePass
Status IRNodePass::Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  if (root_ir == nullptr || modified == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, "Null pointer passed to NodePass");
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
#ifndef ENABLE_ANDROID
Status IRNodePass::Visit(std::shared_ptr<CacheLookupNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<CacheLookupNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<CacheMergeNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<CacheMergeNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::Visit(std::shared_ptr<CacheNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<DatasetNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<CacheNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<DatasetNode>(node), modified);
}
#endif
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
Status IRNodePass::Visit(std::shared_ptr<TFRecordNode> node, bool *const modified) {
  return Visit(std::static_pointer_cast<NonMappableSourceNode>(node), modified);
}
Status IRNodePass::VisitAfter(std::shared_ptr<TFRecordNode> node, bool *const modified) {
  return VisitAfter(std::static_pointer_cast<NonMappableSourceNode>(node), modified);
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
}  // namespace dataset
}  // namespace mindspore
