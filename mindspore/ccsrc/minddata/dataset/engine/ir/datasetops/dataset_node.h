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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_DATASET_NODE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_DATASET_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/filter_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/datasetops/project_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/engine/ir/cache/dataset_cache.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class Dataset;
class DatasetCache;
class SamplerObj;
class IRNodePass;
class DatasetSizeGetter;

// Names for non-leaf IR node
constexpr char kBatchNode[] = "Batch";
constexpr char kBucketBatchByLengthNode[] = "BucketBatchByLength";
constexpr char kBuildSentencePieceVocabNode[] = "BuildSentencePieceVocab";
constexpr char kBuildVocabNode[] = "BuildVocab";
constexpr char kCacheLookupNode[] = "CacheLookup";
constexpr char kCacheMergeNode[] = "CacheMerge";
constexpr char kCacheNode[] = "Cache";
constexpr char kConcatNode[] = "Concat";
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
  // Allow DeepCopyPass to access internal members
  friend class DeepCopyPass;

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

  /// \brief Pure virtual function to clone a new copy of the node
  /// \return The new copy of the node
  virtual std::shared_ptr<DatasetNode> Copy() = 0;

  /// \brief Print the IR tree to output stream
  /// \param out - The output stream to write output to
  void PrintTree(std::ostream &out) const;

  /// \brief << Stream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param out - reference to the output stream being overloaded
  /// \param node - reference to the DatasetNode to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const DatasetNode &node) {
    node.PrintTree(out);
    return out;
  }

  /// \brief Pure virtual function to convert a DatasetNode class into a runtime dataset object
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  virtual Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) = 0;

  /// \brief base virtual function for derived class to implement parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  virtual Status ValidateParams();

  /// \brief Pure virtual function for derived class to get the shard id of specific node
  /// \return Status Status::OK() if get shard id successfully
  virtual Status GetShardId(int32_t *const shard_id);

  /// \brief Gets the dataset size
  /// \param[in] size_getter Shared pointer to DatasetSizeGetter
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \return Status - The status code return
  virtual Status GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                int64_t *dataset_size);

  /// \brief Getter function for child nodes
  /// \return Child nodes
  const std::vector<std::shared_ptr<DatasetNode>> Children() const { return children_; }

  /// \brief Establish a parent-child relationship between this node and the input node.
  ///    Used during the cloning of the user-input IR tree (temporary use)
  Status AppendChild(std::shared_ptr<DatasetNode> child);

  /// \brief Insert the input <node> above this node
  Status InsertAbove(std::shared_ptr<DatasetNode> node);

  /// \brief Add the input node as the next sibling (future use)
  Status InsertChildAt(int32_t pos, std::shared_ptr<DatasetNode> node);

  /// \brief detach this node from its parent, add its child (if any) to its parent
  /// \return error code, return error if node has more than 1 children
  Status Drop();

  /// \brief Check if this node has cache
  /// \return True if the data of this node will be cached
  const bool IsCached() const { return (cache_ != nullptr); }

  /// \brief Check if this node is a leaf node.
  /// \return True if this is a leaf node.
  const bool IsLeaf() const { return children_.empty(); }

  /// \brief Check if this node is a unary operator node.
  /// \return True if this node is semantically a unary operator node
  const bool IsUnaryOperator() const { return (mappable_ == kNotADataSource && !nary_op_); }

  /// \brief Check if this node is a n-ary operator node.
  /// \return True if this node is semantically a n-ary operator node
  const bool IsNaryOperator() const { return (mappable_ == kNotADataSource && nary_op_); }

  /// \brief Check if this node is a mappable dataset. Only applicable to leaf nodes
  /// \return True if this node is a mappable dataset
  const bool IsMappableDataSource() const { return (mappable_ == kMappableSource); }

  /// \brief Check if this node is a non-mappable dataset. Only applicable to leaf nodes
  /// \return True if this node is a non-mappable dataset
  const bool IsNonMappableDataSource() const { return (mappable_ == kNonMappableSource); }

  /// \brief Check if this node is a data source node.
  /// \return True if this node is a data source node
  const bool IsDataSource() const { return (mappable_ == kMappableSource || mappable_ == kNonMappableSource); }

  /// \brief Check if this node is not a data source node.
  /// \return True if this node is not a data source node
  const bool IsNotADataSource() const { return (mappable_ == kNotADataSource); }

  /// \brief Check if this node is a descendant of an operator with cache. Currently used in leaf nodes
  /// \return True if a cache-enabled operator is an ancestor of this node
  const bool IsDescendantOfCache() const { return descendant_of_cache_; }

  /// \brief Check if this node is an orphan node
  /// \return True if this node isn't nullptr nor does it have any children and a parent
  static bool IsOrphanNode(std::shared_ptr<DatasetNode> node) {
    return node != nullptr && node->parent_ == nullptr && node->Children().empty();
  }

  /// \brief Mark to indicate this node is a descendant of an operator with cache. Currently used in leaf nodes
  void HasCacheAbove() { descendant_of_cache_ = true; }

  /// \brief Getter of the number of workers
  int32_t num_workers() { return num_workers_; }

  /// \brief Getter of dataset cache
  std::shared_ptr<DatasetCache> GetDatasetCache() { return cache_; }

  /// \brief Setter function for runtime number of workers
  /// \param[in] num_workers The number of threads in this operator
  /// \return Shared pointer to the original object
  std::shared_ptr<DatasetNode> SetNumWorkers(int32_t num_workers);

  /// \brief Setter function for DatasetCache
  /// \param[in] cache Shared pointer to DatasetCache
  /// \return Shared pointer to the original object
  std::shared_ptr<DatasetNode> SetDatasetCache(const std::shared_ptr<DatasetCache> &cache);

  /// \brief A helper templated function for casting "this" pointer to shared_ptr<derived>
  ///     Similar to shared_from_this, except this one will give you the derived class as shared_ptr
  /// \return A shared_ptr casted to the derived class
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }

  /// \brief Base method for IRNodePass visit. A tree walk consists of walking down the tree and also walking back up
  ///     in a depth-first order. Accept is the node visit on the way down, whereas AcceptAfter is the node
  ///     visit on the way back up the tree after its descendants are visited.
  /// \notes Subclass needs to override this if it requires special node visit access.
  ///     Check "dataset/engine/opt/pass.h" for more details.
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  virtual Status Accept(IRNodePass *const p, bool *const modified);

  /// \brief Base method for IRNodePass visit on the way back up the tree after its descendants are visited.
  /// \notes Subclass needs to override this if it requires special node visit access.
  ///     Check "dataset/engine/opt/pass.h" for more details.
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  virtual Status AcceptAfter(IRNodePass *const p, bool *const modified);

  virtual bool IsSizeDefined() { return true; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  virtual Status to_json(nlohmann::json *out_json);

  /// \brief Setter function, set the number of total repeats for the operator
  void SetTotalRepeats(int32_t total_repeats) { total_repeats_ = total_repeats; }

  /// \brief Setter function, set the number of epochs for the operator
  void SetNumEpochs(int32_t num_epochs) { num_epochs_ = num_epochs; }

  /// \brief Getter function
  /// \return The number of required repeats for the operator
  int32_t GetTotalRepeats() const { return total_repeats_; }

  /// \brief Getter function
  /// \return The number of epochs for the operator
  int32_t GetNumEpochs() const { return num_epochs_; }

  /// \brief Getter function
  /// \return The number of repeats per epoch for the operator
  int32_t GetNumRepeatsPerEpoch() const { return total_repeats_ / num_epochs_; }

 protected:
  std::vector<std::shared_ptr<DatasetNode>> children_;
  DatasetNode *parent_;  // used to record the only one parent of an IR node after parsing phase
  std::shared_ptr<DatasetCache> cache_;
  int64_t dataset_size_;
  int32_t num_workers_;
  int32_t rows_per_buffer_;
  int32_t connector_que_size_;
  int32_t worker_connector_size_;
  int32_t total_repeats_;  // Number of times required to run this operator
  int32_t num_epochs_;     // Number of epochs
  // Establish a parent-child relationship between this node and the input node.
  // Used only in the constructor of the class and its derived classes.
  void AddChild(std::shared_ptr<DatasetNode> child);
  std::string PrintColumns(const std::vector<std::string> &columns) const;
  void PrintNode(std::ostream &out, int *level) const;
  enum DataSource { kNotADataSource = 0, kNonMappableSource = 1, kMappableSource = 2 };
  enum DataSource mappable_;
  bool nary_op_;  // an indicator of whether the current node supports multiple children, true for concat/zip node
  bool descendant_of_cache_;
};

// MappableSourceNode represents the leaf nodes that can be randomly accessed with indexes.
class MappableSourceNode : public DatasetNode {
 public:
  /// \brief Constructor
  MappableSourceNode() : DatasetNode() { mappable_ = kMappableSource; }

  /// \brief Constructor that initializes the cache
  /// \param dataset_cache DatasetCache
  explicit MappableSourceNode(const std::shared_ptr<DatasetCache> &dataset_cache) : DatasetNode(dataset_cache) {
    mappable_ = kMappableSource;
    // Initially set to false, and set to true by the optimizer when conditions are met.
    descendant_of_cache_ = false;
  }

  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Destructor
  ~MappableSourceNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;

  /// \brief Sampler getter
  /// \return SamplerObj of the current node
  virtual std::shared_ptr<SamplerObj> Sampler() = 0;

  /// \brief Sampler setter
  virtual void SetSampler(std::shared_ptr<SamplerObj> sampler) = 0;
};

// NonMappableSourceNode represents the leaf nodes that can not be randomly accessed.
class NonMappableSourceNode : public DatasetNode {
 public:
  /// \brief Constructor
  NonMappableSourceNode() : DatasetNode() { mappable_ = kNonMappableSource; }

  /// \brief Constructor that initializes the cache
  /// \param dataset_cache DatasetCache
  explicit NonMappableSourceNode(const std::shared_ptr<DatasetCache> &dataset_cache) : DatasetNode(dataset_cache) {
    mappable_ = kNonMappableSource;
    // Initially set to false, and set to true by the optimizer when conditions are met.
    descendant_of_cache_ = false;
  }

  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Destructor
  ~NonMappableSourceNode() = default;

  /// \brief Node name getter
  /// \return Name of the current node
  virtual std::string Name() const = 0;

  /// \brief By default non-mappable dataset does not support sampling. However, if a cache operator
  ///     is injected at some other place higher in the tree, that cache can inherit this sampler
  ///     from the leaf, providing sampling support from the caching layer.
  ///     This function sets up the sampler for a leaf node that does not use sampling.
  /// \param[in] sampler The sampler to setup
  /// \return Status of the function
  virtual Status SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) = 0;

  /// \brief If a cache has been added into the ascendant tree over this non-mappable source node, then the cache will
  ///     be executing a sampler for fetching the data. As such, any options in the source node need to be reset to its
  ///     defaults so that this source node will produce the full set of data into the cache.
  /// \return Status of the function
  virtual Status MakeSimpleProducer() = 0;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_DATASETOPS_DATASET_NODE_H_
