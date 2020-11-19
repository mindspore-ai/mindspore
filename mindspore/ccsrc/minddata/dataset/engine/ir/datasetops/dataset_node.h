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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_DATASET_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_DATASET_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {

class Dataset;
class SamplerObj;
class NodePass;

#define RETURN_EMPTY_IF_ERROR(_s) \
  do {                            \
    Status __rc = (_s);           \
    if (__rc.IsError()) {         \
      MS_LOG(ERROR) << __rc;      \
      return {};                  \
    }                             \
  } while (false)

// Names for non-leaf IR node
constexpr char kBatchNode[] = "Batch";
constexpr char kBucketBatchByLengthNode[] = "BucketBatchByLength";
constexpr char kBuildSentencePieceVocabNode[] = "BuildSentencePieceVocab";
constexpr char kBuildVocabNode[] = "BuildVocab";
constexpr char kConcatNode[] = "Concat";
constexpr char kDatasetNode[] = "Dataset";
constexpr char kEpochCtrlNode[] = "EpochCtrl";
constexpr char kFilterNode[] = "Filter";
constexpr char kMapNode[] = "Map";
constexpr char kProjectNode[] = "Project";
constexpr char kRenameNode[] = "Rename";
constexpr char kRepeatNode[] = "Repeat";
constexpr char kRootNode[] = "Top";
constexpr char kShuffleNode[] = "Shuffle";
constexpr char kSkipNode[] = "Skip";
constexpr char kSyncWaitNode[] = "SyncWait";
constexpr char kTakeNode[] = "Take";
constexpr char kTransferNode[] = "Transfer";
constexpr char kZipNode[] = "Zip";

// Names for leaf IR node
constexpr char kAlbumNode[] = "AlbumDataset";
constexpr char kCelebANode[] = "CelebADataset";
constexpr char kCifar100Node[] = "Cifar100Dataset";
constexpr char kCifar10Node[] = "Cifar10Dataset";
constexpr char kCLUENode[] = "CLUEDataset";
constexpr char kCocoNode[] = "CocoDataset";
constexpr char kCSVNode[] = "CSVDataset";
constexpr char kGeneratorNode[] = "GeneratorDataset";
constexpr char kImageFolderNode[] = "ImageFolderDataset";
constexpr char kManifestNode[] = "ManifestDataset";
constexpr char kMindDataNode[] = "MindDataDataset";
constexpr char kMnistNode[] = "MnistDataset";
constexpr char kRandomNode[] = "RandomDataset";
constexpr char kTextFileNode[] = "TextFileDataset";
constexpr char kTFRecordNode[] = "TFRecordDataset";
constexpr char kVOCNode[] = "VOCDataset";

Status AddShuffleOp(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                    int32_t connector_que_size, int32_t rows_per_buffer, std::shared_ptr<DatasetOp> *shuffle_op);

// Helper function to validate dataset files parameter
Status ValidateDatasetFilesParam(const std::string &dataset_name, const std::vector<std::string> &dataset_files);

// Helper function to validate dataset num_shards and shard_id parameters
Status ValidateDatasetShardParams(const std::string &dataset_name, int32_t num_shards, int32_t shard_id);

// Helper function to validate dataset sampler parameter
Status ValidateDatasetSampler(const std::string &dataset_name, const std::shared_ptr<SamplerObj> &sampler);

Status ValidateStringValue(const std::string &dataset_name, const std::string &str,
                           const std::unordered_set<std::string> &valid_strings);

// Helper function to validate dataset input/output column parameterCD -
Status ValidateDatasetColumnParam(const std::string &dataset_name, const std::string &column_param,
                                  const std::vector<std::string> &columns);

// Helper function to validate dataset directory parameter
Status ValidateDatasetDirParam(const std::string &dataset_name, std::string dataset_dir);

/// \brief Function to create a sampler for non-mappable dataset (to be used by cache op later).
/// \notes Non-mappable dataset does not directly support a sampler. It has provided sampling arguments (shuffle,
///     num_samples, num_shards, shard_id) and it DOES support sampling if somewhere above it in the pipeline contains
///     a cache. If there is no cache above it, then the sampler is not used.
/// \param[in] num_samples The number of samples to be included in the dataset.
/// \param[in] shuffle If true, the indices are shuffled.
/// \param[in] num_shards Number of shards to divide the dataset into.
/// \param[in] shard_id Shard ID of the current shard within num_shards.
/// \return Shared pointer to the current Sampler.
std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards, int32_t shard_id);

// The base class of all IR nodes
class DatasetNode : public std::enable_shared_from_this<DatasetNode> {
 public:
  /// \brief Constructor
  DatasetNode();

  /// \brief Constructor that initializes the cache
  /// \param dataset_cache DatasetCache
  explicit DatasetNode(const std::shared_ptr<DatasetCache> &dataset_cache);

  /// \brief Destructor
  ~DatasetNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;

  /// \brief Pure virtual function to print the description
  /// \param out - The output stream to write output to
  virtual void Print(std::ostream &out) const = 0;

  /// \brief Pure virtual function to make a new copy of the node
  /// \return The new copy of the node
  virtual std::shared_ptr<DatasetNode> Copy() = 0;

  /// \brief Print the IR tree to output stream
  /// \param out - The output stream to write output to
  void PrintTree(std::ostream &out) const;

  /// \brief << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out - reference to the output stream being overloaded
  /// \param dO - reference to the DatasetOp to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DatasetNode &node) {
    node.PrintTree(out);
    return out;
  }

  /// \brief Make a new copy of the tree from the current node
  /// \return The new copy of the tree
  std::shared_ptr<DatasetNode> DeepCopy();

  /// \brief Pure virtual function to convert a DatasetNode class into a runtime dataset object
  /// \return The list of shared pointers to the newly created DatasetOps
  virtual std::vector<std::shared_ptr<DatasetOp>> Build() = 0;

  /// \brief Pure virtual function for derived class to implement parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  virtual Status ValidateParams() = 0;

  /// \brief Pure virtual function for derived class to get the shard id of specific node
  /// \return Status Status::OK() if get shard id successfully
  virtual Status GetShardId(int32_t *shard_id);

  /// \brief Getter function for child nodes
  /// \return Child nodes
  const std::vector<std::shared_ptr<DatasetNode>> Children() const { return children_; }

  /// \brief Establish the parent-child relationship between this node and its child.
  void AddChild(std::shared_ptr<DatasetNode> child);

  /// \brief detach this node from its parent, add its child (if any) to its parent
  /// \return error code, return error if node has more than 1 children
  Status Remove();

  /// \brief  Check if this node has cache
  /// \return True if the data of this node will be cached
  const bool IsCached() const { return (cache_ != nullptr); }

  /// \brief Setter function for runtime number of workers
  /// \param[in] num_workers The number of threads in this operator
  /// \return Shared pointer to the original object
  std::shared_ptr<DatasetNode> SetNumWorkers(int32_t num_workers);

  /// \brief A helper templated function for casting "this" pointer to shared_ptr<derived>
  ///     Similar to shared_from_this, except this one will give you the derived class as shared_ptr
  /// \return A shared_ptr casted to the derived class
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }

  /// \brief Base method for NodePass visit. A tree walk consists of walking down the tree and also walking back up
  ///     in a depth-first order. Accept is the node visit on the way down, whereas AcceptAfter is the node
  ///     visit on the way back up the tree after its descendants are visited.
  /// \notes Subclass needs to override this if it requires special node visit access.
  ///     Check "dataset/engine/opt/pass.h" for more details.
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  virtual Status Accept(NodePass *p, bool *modified);

  /// \brief Base method for NodePass visit on the way back up the tree after its descendants are visited.
  /// \notes Subclass needs to override this if it requires special node visit access.
  ///     Check "dataset/engine/opt/pass.h" for more details.
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  virtual Status AcceptAfter(NodePass *p, bool *modified);

  /// \brief Method to get status from Node.Build()
  /// \notes Remove me after changing return val of Build()
  Status BuildStatus() { return build_status; }

 protected:
  std::vector<std::shared_ptr<DatasetNode>> children_;
  DatasetNode *parent_;
  std::shared_ptr<DatasetCache> cache_;
  int32_t num_workers_;
  int32_t rows_per_buffer_;
  int32_t connector_que_size_;
  int32_t worker_connector_size_;
  Status build_status;  // remove me after changing return val of Build()
  std::string PrintColumns(const std::vector<std::string> &columns) const;
  Status AddCacheOp(std::vector<std::shared_ptr<DatasetOp>> *node_ops);
  void PrintNode(std::ostream &out, int *level) const;
};

// SourceNode represents the leaf nodes of a pipeline where the data is pulled into.
class SourceNode : public DatasetNode {
 public:
  /// \brief Constructor
  SourceNode() : DatasetNode() {}

  /// \brief Constructor that initializes the cache
  /// \param dataset_cache DatasetCache
  explicit SourceNode(const std::shared_ptr<DatasetCache> &dataset_cache) : DatasetNode(dataset_cache) {}

  /// \brief Destructor
  ~SourceNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;

  /// \brief Base-class override for accepting NodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(NodePass *p, bool *modified) override;

  /// \brief Base-class override for accepting NodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(NodePass *p, bool *modified) override;

  /// \brief  Check if this node is a mappable dataset. Only applicable to leaf nodes
  /// \return True if the dataset represented by this node is a mappable dataset
  const bool IsMappable() const { return mappable_; }

 protected:
  bool mappable_;
};

// MappableSourceNode represents the leaf nodes that can be randomly accessed with indexes.
class MappableSourceNode : public SourceNode {
 public:
  /// \brief Constructor
  MappableSourceNode() : SourceNode() { mappable_ = true; }

  /// \brief Constructor that initializes the cache
  /// \param dataset_cache DatasetCache
  explicit MappableSourceNode(const std::shared_ptr<DatasetCache> &dataset_cache) : SourceNode(dataset_cache) {
    mappable_ = true;
  }

  /// \brief Destructor
  ~MappableSourceNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;
};

// NonMappableSourceNode represents the leaf nodes that can not be randomly accessed.
class NonMappableSourceNode : public SourceNode {
 public:
  /// \brief Constructor
  NonMappableSourceNode() : SourceNode() { mappable_ = false; }

  /// \brief Constructor that initializes the cache
  /// \param dataset_cache DatasetCache
  explicit NonMappableSourceNode(const std::shared_ptr<DatasetCache> &dataset_cache) : SourceNode(dataset_cache) {
    mappable_ = false;
  }

  /// \brief Destructor
  ~NonMappableSourceNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;
};

// NonLeafNode represents operations over data in a pipeline.
class NonLeafNode : public DatasetNode {
 public:
  /// \brief Constructor
  NonLeafNode() = default;

  /// \brief Destructor
  ~NonLeafNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;
};

// SinkNode represents the end node of a pipeline where the data is pushed out
class SinkNode : public DatasetNode {
 public:
  /// \brief Constructor
  SinkNode() = default;

  /// \brief Destructor
  ~SinkNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_DATASET_NODE_H_
