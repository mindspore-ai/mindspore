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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_H_

#include <memory>
#include <queue>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Non-leaf IR node
class BatchNode;
class BucketBatchByLengthNode;
class BuildVocabNode;
#ifndef ENABLE_ANDROID
class CacheLookupNode;
class CacheMergeNode;
class CacheNode;
#endif
class ConcatNode;
class EpochCtrlNode;
class FilterNode;
class MapNode;
class ProjectNode;
class RenameNode;
class RepeatNode;
class RootNode;
class ShuffleNode;
class SkipNode;
class TakeNode;
class TFRecordNode;
class TransferNode;
class ZipNode;
#ifdef ENABLE_PYTHON
class SyncWaitNode;
#endif
#ifndef ENABLE_ANDROID
class BuildSentenceVocabNode;
#endif
// Leaf IR node
class AlbumNode;
class CelebANode;
class Cifar100Node;
class Cifar10Node;
class CocoNode;
class ImageFolderNode;
class ManifestNode;
class MnistNode;
class RandomNode;
class VOCNode;
#ifdef ENABLE_PYTHON
class GeneratorNode;
#endif
#ifndef ENABLE_ANDROID
class CLUENode;
class CSVNode;
class MindDataNode;
class TextFileNode;
class TFRecordNode;
#endif

// The base class Pass is the basic unit of tree transformation.
// The actual implementation of the passes will be derived from here.
class IRPass : public std::enable_shared_from_this<IRPass> {
 public:
  // Run the transformation pass against the IR tree.
  // @param root_ir - Pointer to the IR tree to be transformed.
  // @param modified - Pointer to the modified flag,
  virtual Status Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) = 0;

  virtual ~IRPass() = default;
};

// IRTreePass is a basic Pass class which performs transformation on IR tree directly.
class IRTreePass : public IRPass {
 public:
  /// \brief Run the transformation pass against the IR tree.
  /// \param[in,out] root_ir Pointer to the IR tree to be transformed.
  /// \param[in,out] modified Indicate if the tree was modified
  Status Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) final;

  /// \brief Derived classes may implement the runOnTree function to implement tree transformation.
  ///     "modified" flag needs to be set to true if tree is modified during the pass execution.
  /// \param[in,out] tree The tree to operate on.
  /// \param[in,out] Indicate if the tree was modified.
  /// \return Status The status code returned
  virtual Status RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) { return Status::OK(); }
};

// IRNodePass is a base Pass class which performs transformation on node visiting.
// IRNodePass implements Visitor design pattern.
// The visiting happens twice for each node in the DFS traversal, one on the way down of the traversal,
// and the other when all the descending nodes are visited.
// Actual transformation is done by implementing a new derived class of IRNodePass.
// The derived class will implement the method Visit()/VisitAfter() passing specified node types
// it wants to action on them, overriding the ones defined in IRNodePass.
// If the derived class wants to perform the same action on all node types,
// it can simply implement the method Visit()/VisitAfter() passing the base class DatasetNode.
// This is made possible by overloading the method Visit()/VisitAfter() on each node type to fall back
// to call the Visit()/VisitAfter() in this parent IRNodePass class.
class IRNodePass : public IRPass {
 public:
  // Tree traversal order
  enum Order { DFS, BFS };

  // Constructor
  // Default DFS traversal
  explicit IRNodePass(Order order = Order::DFS) { traversalOrder_ = order; }

  ~IRNodePass() = default;

  /// \brief Run the transformation pass against the IR tree
  /// \param[in,out] root_ir Pointer to the IR tree to be transformed
  /// \param[in,out] modified Indicator if the tree was changed
  Status Run(std::shared_ptr<DatasetNode> root_ir, bool *const modified) final;

  /// \brief Derived classes may implement the Visit function to implement any initial visit work on the way down
  ///     a tree traversal.  "modified" flag needs to be set to true if node is modified during the pass execution
  /// \param[in] node The node being visited
  /// \param[out] modified Indicator if the node was changed at all
  /// \return Status The status code returned
  virtual Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) { return Status::OK(); }

  /// \brief Derived classes may implement the VisitAfter function to implement node level tree transformation
  ///     "modified" flag needs to be set to true if node is modified during the pass execution
  /// \param[in] node The node being visited
  /// \param[out] modified Indicator if the node was changed at all.
  /// \return Status The status code returned
  virtual Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) { return Status::OK(); }

  // Visit()/VisitAfter() method to be overridden.
  // These pairs of Visit()/VisitAfter() for each derived class of DatasetNode are defined here.
  // Their implementation are in .cc file to avoid adding the include files of those derived classes.
  // The implementation simply falls back to call Visit()/VisitAfter of class DatasetNode, the parent of
  // the derived classes. With this technique, the transformation classes derived from NodePass needs only to
  // implement Visit()/VisitAfter() passing DatasetNode if it wants to action on any derived classes
  // of DatasetNode in the same way.
  // Note that virtual template functions are not permitted in C++.
  //
  // Non-leaf IR node
  virtual Status Visit(std::shared_ptr<BatchNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<BatchNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<BucketBatchByLengthNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<BucketBatchByLengthNode> node, bool *const modified);
#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<BuildSentenceVocabNode> node, bool *const modified);
#endif
  virtual Status Visit(std::shared_ptr<BuildVocabNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<BuildVocabNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<ConcatNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<ConcatNode> node, bool *const modified);
#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<CacheMergeNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<CacheMergeNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<CacheLookupNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<CacheLookupNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<CacheNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<CacheNode> node, bool *const modified);
#endif
  virtual Status Visit(std::shared_ptr<EpochCtrlNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<EpochCtrlNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<FilterNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<FilterNode> node, bool *const modified);
#ifdef ENABLE_PYTHON
  virtual Status Visit(std::shared_ptr<GeneratorNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<GeneratorNode> node, bool *const modified);
#endif
  virtual Status Visit(std::shared_ptr<MapNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<MapNode> node, bool *const modified);
#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<MindDataNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<MindDataNode> node, bool *const modified);
#endif
  virtual Status Visit(std::shared_ptr<ProjectNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<ProjectNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<RandomNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<RandomNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<RenameNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<RenameNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<RepeatNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<RootNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<RootNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<ShuffleNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<ShuffleNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<SkipNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<SkipNode> node, bool *const modified);
#ifdef ENABLE_PYTHON
  virtual Status Visit(std::shared_ptr<SyncWaitNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<SyncWaitNode> node, bool *const modified);
#endif
  virtual Status Visit(std::shared_ptr<TakeNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<TakeNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<TFRecordNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<TFRecordNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<TransferNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<TransferNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<ZipNode> node, bool *const modified);
  virtual Status VisitAfter(std::shared_ptr<ZipNode> node, bool *const modified);

  // leaf-IR Node
  virtual Status Visit(std::shared_ptr<MappableSourceNode> node, bool *const modified);
  virtual Status Visit(std::shared_ptr<NonMappableSourceNode> node, bool *const modified);

 private:
  // Helper function to perform DFS visit
  Status DFSNodeVisit(std::shared_ptr<DatasetNode> node_ir, bool *const modified);

  // Helper function to perform BFS visit
  Status BFSNodeVisit(std::shared_ptr<DatasetNode> node_ir, bool *const modified);

  // Tree traversal order of the NodePass
  Order traversalOrder_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_H_
