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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_H_

#include <memory>
#include <queue>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class BatchNode;
class BucketBatchByLengthNode;
#ifndef ENABLE_ANDROID
class BuildSentenceVocabNode;
#endif
class BuildVocabNode;
class ConcatNode;
class MapNode;
class ProjectNode;
class RenameNode;
class RepeatNode;
class ShuffleNode;
class SkipNode;
#ifdef ENABLE_PYTHON
class SyncWaitNode;
#endif
class TakeNode;
class TransferNode;
class ZipNode;
class AlbumNode;
class CelebANode;
class Cifar100Node;
class Cifar10Node;
#ifndef ENABLE_ANDROID
class CLUENode;
#endif
class CocoNode;
#ifndef ENABLE_ANDROID
class CSVNode;
#endif
#ifdef ENABLE_PYTHON
class GeneratorNode;
#endif
class ImageFolderNode;
class ManifestNode;
#ifndef ENABLE_ANDROID
class MindDataNode;
#endif
class MnistNode;
class RandomNode;
#ifndef ENABLE_ANDROID
class TextFileNode;
#endif
#ifndef ENABLE_ANDROID
class TFRecordNode;
#endif
class VOCNode;

//////////////////////////////////
// This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
class BatchOp;

class MapOp;

class ProjectOp;

class RenameOp;

class SkipOp;

class ShuffleOp;

class AlbumOp;

class RandomDataOp;

class RepeatOp;

class TakeOp;

class ZipOp;

class DeviceQueueOp;

class ImageFolderOp;

class MnistOp;

class ManifestOp;

class CifarOp;

class VOCOp;

class CocoOp;

class CelebAOp;

class EpochCtrlOp;

class BuildVocabOp;

class ConcatOp;

#ifndef ENABLE_ANDROID
class MindRecordOp;

class TFReaderOp;

class CacheOp;

class CacheMergeOp;

class CacheLookupOp;

class BuildSentencePieceVocabOp;

class ClueOp;

class CsvOp;

class TextFileOp;
#endif

#ifdef ENABLE_PYTHON
class FilterOp;

class GeneratorOp;
#endif
//////////////////////////////////

// The base class Pass is the basic unit of tree transformation.
// The actual implementation of the passes will be derived from here.
class Pass : public std::enable_shared_from_this<Pass> {
 public:
  // Run the transformation pass against the IR tree.
  // @param root_ir - Pointer to the IR tree to be transformed.
  // @param modified - Pointer to the modified flag,
  virtual Status Run(std::shared_ptr<DatasetNode> root_ir, bool *modified) = 0;

  //////////////////////////////////
  // This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
  // Run the transformation pass against the execution tree.
  // @param tree - Pointer to the execution tree to be transformed.
  // @param modified - Pointer to the modified flag,
  virtual Status Run(ExecutionTree *tree, bool *modified) = 0;
  //////////////////////////////////

  virtual ~Pass() = default;
};

// TreePass is a basic Pass class which performs transformation on ExecutionTree directly.
class TreePass : public Pass {
 public:
  /// \brief Run the transformation pass against the IR tree.
  /// \param[inout] root_ir Pointer to the IR tree to be transformed.
  /// \param[inout] modified Indicate if the tree was modified
  Status Run(std::shared_ptr<DatasetNode> root_ir, bool *modified) final;

  //////////////////////////////////
  // This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
  /// \brief Run the transformation pass against the execution tree.
  /// \param[inout] tree Pointer to the execution tree to be transformed.
  /// \param[inout] modified Indicate if the tree was modified
  Status Run(ExecutionTree *tree, bool *modified) final;

  /// \brief Derived classes may implement the runOnTree function to implement tree transformation.
  ///     "modified" flag needs to be set to true if tree is modified during the pass execution.
  /// \param[inout] tree The tree to operate on.
  /// \param[inout] Indicate of the tree was modified.
  /// \return Status The error code return
  virtual Status RunOnTree(ExecutionTree *tree, bool *modified) { return Status::OK(); }
  //////////////////////////////////
};

// NodePass is a basic Pass class which performs transformation on Node visiting.
// NodePass implements Visitor design pattern.
class NodePass : public Pass {
 public:
  // Tree traversal order
  enum Order { DFS, BFS };

  // Constructor
  // Default DFS traversal
  explicit NodePass(Order order = Order::DFS) { traversalOrder_ = order; }

  ~NodePass() = default;

  /// \brief Run the transformation pass against the IR tree
  /// \param[inout] root_ir Pointer to the IR tree to be transformed
  /// \param[inout] modified Indicator if the tree was changed
  Status Run(std::shared_ptr<DatasetNode> root_ir, bool *modified) final;

  /// \brief Derived classes may implement the Visit function to implement any initial visit work on the way down
  ///     a tree traversal.  "modified" flag needs to be set to true if node is modified during the pass execution
  /// \param[in] node The node being visited
  /// \param[out] modified Indicator if the node was changed at all
  /// \return Status The error code return
  virtual Status Visit(std::shared_ptr<DatasetNode> node, bool *modified) { return Status::OK(); }

  /// \brief Derived classes may implement the VisitAfter function to implement node level tree transformation
  ///     "modified" flag needs to be set to true if node is modified during the pass execution
  /// \param[in] node The node being visited
  /// \param[out] modified Indicator if the node was changed at all.
  /// \return Status The error code return
  virtual Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *modified) { return Status::OK(); }

  // For datasetops IR
  // Visit method to be overridden.
  // Note that member template can not be virtual, any node which wants to work with NodePass
  // should declare Visit of its own type and override "Accept" from DatasetNode.
  virtual Status Visit(std::shared_ptr<BatchNode> node, bool *modified);

  // VisitAfter method to be overridden.
  // Note that member template can not be virtual, any node which wants to work with NodePass
  // should declare VisitAfter of its own type and override "AcceptAfter" from DatasetNode.
  virtual Status VisitAfter(std::shared_ptr<BatchNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<BucketBatchByLengthNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<BucketBatchByLengthNode> node, bool *modified);

#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<BuildSentenceVocabNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<BuildSentenceVocabNode> node, bool *modified);
#endif

  virtual Status Visit(std::shared_ptr<BuildVocabNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<BuildVocabNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<ConcatNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<ConcatNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<MapNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<MapNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<ProjectNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<ProjectNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<RenameNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<RenameNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<RepeatNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<RepeatNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<ShuffleNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<ShuffleNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<SkipNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<SkipNode> node, bool *modified);

#ifdef ENABLE_PYTHON
  virtual Status Visit(std::shared_ptr<SyncWaitNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<SyncWaitNode> node, bool *modified);
#endif

  virtual Status Visit(std::shared_ptr<TakeNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<TakeNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<TransferNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<TransferNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<ZipNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<ZipNode> node, bool *modified);

  // For datasetops/source IR
  virtual Status Visit(std::shared_ptr<AlbumNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<AlbumNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<CelebANode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<CelebANode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<Cifar100Node> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<Cifar100Node> node, bool *modified);

  virtual Status Visit(std::shared_ptr<Cifar10Node> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<Cifar10Node> node, bool *modified);

#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<CLUENode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<CLUENode> node, bool *modified);
#endif

  virtual Status Visit(std::shared_ptr<CocoNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<CocoNode> node, bool *modified);

#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<CSVNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<CSVNode> node, bool *modified);
#endif

#ifdef ENABLE_PYTHON
  virtual Status Visit(std::shared_ptr<GeneratorNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<GeneratorNode> node, bool *modified);
#endif

  virtual Status Visit(std::shared_ptr<ImageFolderNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<ImageFolderNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<ManifestNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<ManifestNode> node, bool *modified);

#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<MindDataNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<MindDataNode> node, bool *modified);
#endif

  virtual Status Visit(std::shared_ptr<MnistNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<MnistNode> node, bool *modified);

  virtual Status Visit(std::shared_ptr<RandomNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<RandomNode> node, bool *modified);

#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<TextFileNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<TextFileNode> node, bool *modified);
#endif

#ifndef ENABLE_ANDROID
  virtual Status Visit(std::shared_ptr<TFRecordNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<TFRecordNode> node, bool *modified);
#endif

  virtual Status Visit(std::shared_ptr<VOCNode> node, bool *modified);

  virtual Status VisitAfter(std::shared_ptr<VOCNode> node, bool *modified);

  //////////////////////////////////
  // This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
  /// \brief Run the transformation pass against the execution tree
  /// \param[inout] tree Pointer to the execution tree to be transformed
  /// \param[inout] modified Indicator if the tree was changed
  Status Run(ExecutionTree *tree, bool *modified) final;

  /// \brief Derived classes may implement the PreRunOnNode function to implement any initial visit work on the way down
  ///     a tree traversal.  "modified" flag needs to be set to true if tree is modified during the pass execution
  /// \param[in] node The node being visited
  /// \param[out] modified Indicator if the node was changed at all
  /// \return Status The error code return
  virtual Status PreRunOnNode(std::shared_ptr<DatasetOp> node, bool *modified) { return Status::OK(); }

  /// \brief Derived classes may implement the RunOnNode function to implement node level tree transformation
  ///     "modified" flag needs to be set to true if tree is modified during the pass execution
  /// \param[in] node The node being visited
  /// \param[out] modified Indicator if the node was changed at all.
  /// \return Status The error code return
  virtual Status RunOnNode(std::shared_ptr<DatasetOp> node, bool *modified) { return Status::OK(); }

  // Visit methods to be overridden.
  // Note that member template can not be virtual, any op which wants to work with NodePass should declare RunOnNode
  // of its own type and override "Accept" from DatasetOp.
  virtual Status RunOnNode(std::shared_ptr<BatchOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<MapOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<ProjectOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<RenameOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<SkipOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<RandomDataOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<AlbumOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<TakeOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<ZipOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<MnistOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CifarOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CocoOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CelebAOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<RepeatOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<RepeatOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<EpochCtrlOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<BuildVocabOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<ZipOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<MapOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<ConcatOp> node, bool *modified);

#ifndef ENABLE_ANDROID
  virtual Status RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<TFReaderOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CacheMergeOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CacheLookupOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CacheOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<ClueOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<CsvOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<TextFileOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<CacheOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<CacheMergeOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<BuildSentencePieceVocabOp> node, bool *modified);
#endif

#ifdef ENABLE_PYTHON
  virtual Status RunOnNode(std::shared_ptr<FilterOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<ManifestOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified);

  virtual Status RunOnNode(std::shared_ptr<VOCOp> node, bool *modified);

  virtual Status PreRunOnNode(std::shared_ptr<FilterOp> node, bool *modified);
#endif
  //////////////////////////////////

 private:
  // Helper function to perform DFS visit
  Status DFSNodeVisit(std::shared_ptr<DatasetNode> node_ir, bool *modified);

  // Helper function to perform BFS visit
  Status BFSNodeVisit(std::shared_ptr<DatasetNode> node_ir, bool *modified);

  //////////////////////////////////
  // This section of code will be removed once the migration of optimizer from DatasetOp to DatasetNode is done.
  // Helper function to perform DFS visit
  Status DFSNodeVisit(std::shared_ptr<DatasetOp> node, bool *modified);

  // Helper function to perform BFS visit
  Status BFSNodeVisit(std::shared_ptr<DatasetOp> root, bool *modified);
  //////////////////////////////////

  // Tree traversal order of the NodePass
  Order traversalOrder_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_H_
