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

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <set>

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {

// Helper function to compute a default shuffle size
Status ComputeShuffleSize(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                          int64_t *shuffle_size) {
  const int64_t average_files_multiplier = 4;
  const int64_t shuffle_max = 10000;
  int64_t avg_rows_per_file = 0;

  // Adjust the num rows per shard if sharding was given
  if (num_devices > 0) {
    if (num_rows % num_devices == 0) {
      num_rows = num_rows / num_devices;
    } else {
      num_rows = (num_rows / num_devices) + 1;
    }
  }

  // Cap based on total rows directive.  Some ops do not have this and give value of 0.
  if (total_rows > 0) {
    num_rows = std::min(num_rows, total_rows);
  }

  // get the average per file
  CHECK_FAIL_RETURN_UNEXPECTED(num_files != 0, "The size of dataset_files must greater than 0.");
  avg_rows_per_file = num_rows / num_files;

  *shuffle_size = std::max(avg_rows_per_file * average_files_multiplier, shuffle_max);
  return Status::OK();
}

// Helper function to inject a shuffle operator over top of current operator being built
Status AddShuffleOp(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                    int32_t connector_que_size, int32_t rows_per_buffer, std::shared_ptr<DatasetOp> *shuffle_op) {
  std::shared_ptr<ShuffleOp> new_shuffle_op = nullptr;
  int64_t shuffle_size = 0;
  RETURN_IF_NOT_OK(ComputeShuffleSize(num_files, num_devices, num_rows, total_rows, &shuffle_size));
  MS_LOG(INFO) << "Dataset::AddShuffleOp - num_rows: " << num_rows << ", shuffle_size: " << shuffle_size;
  // Add the shuffle op
  *shuffle_op = std::make_shared<ShuffleOp>(shuffle_size, GetSeed(), connector_que_size, true, rows_per_buffer);
  return Status::OK();
}

// Helper function to validate dataset directory parameter
Status ValidateDatasetDirParam(const std::string &dataset_name, std::string dataset_dir) {
  if (dataset_dir.empty()) {
    std::string err_msg = dataset_name + ": dataset_dir is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  Path dir(dataset_dir);
  if (!dir.IsDirectory()) {
    std::string err_msg = dataset_name + ": dataset_dir: [" + dataset_dir + "] is an invalid directory path.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (access(dataset_dir.c_str(), R_OK) == -1) {
    std::string err_msg = dataset_name + ": No access to specified dataset path: " + dataset_dir;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Helper function to validate dataset files parameter
Status ValidateDatasetFilesParam(const std::string &dataset_name, const std::vector<std::string> &dataset_files) {
  if (dataset_files.empty()) {
    std::string err_msg = dataset_name + ": dataset_files is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  for (auto f : dataset_files) {
    Path dataset_file(f);
    if (!dataset_file.Exists()) {
      std::string err_msg = dataset_name + ": dataset file: [" + f + "] is invalid or does not exist.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (access(dataset_file.toString().c_str(), R_OK) == -1) {
      std::string err_msg = dataset_name + ": No access to specified dataset file: " + f;
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

// Helper function to validate dataset num_shards and shard_id parameters
Status ValidateDatasetShardParams(const std::string &dataset_name, int32_t num_shards, int32_t shard_id) {
  if (num_shards <= 0) {
    std::string err_msg = dataset_name + ": Invalid num_shards: " + std::to_string(num_shards);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (shard_id < 0 || shard_id >= num_shards) {
    // num_shards;
    std::string err_msg = dataset_name + ": Invalid input, shard_id: " + std::to_string(shard_id) +
                          ", num_shards: " + std::to_string(num_shards);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Helper function to validate dataset sampler parameter
Status ValidateDatasetSampler(const std::string &dataset_name, const std::shared_ptr<SamplerObj> &sampler) {
  if (sampler == nullptr) {
    std::string err_msg = dataset_name + ": Sampler is not constructed correctly, sampler: nullptr";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status ValidateStringValue(const std::string &dataset_name, const std::string &str,
                           const std::unordered_set<std::string> &valid_strings) {
  if (valid_strings.find(str) == valid_strings.end()) {
    std::string mode;
    mode = std::accumulate(valid_strings.begin(), valid_strings.end(), mode,
                           [](std::string a, std::string b) { return std::move(a) + " " + std::move(b); });
    std::string err_msg = dataset_name + ": " + str + " does not match any mode in [" + mode + " ]";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Helper function to validate dataset input/output column parameter
Status ValidateDatasetColumnParam(const std::string &dataset_name, const std::string &column_param,
                                  const std::vector<std::string> &columns) {
  if (columns.empty()) {
    std::string err_msg = dataset_name + ":" + column_param + " should not be empty string";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (uint32_t i = 0; i < columns.size(); ++i) {
    if (columns[i].empty()) {
      std::string err_msg = dataset_name + ":" + column_param + "[" + std::to_string(i) + "] must not be empty";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  std::set<std::string> columns_set;
  for (auto &column_name : columns) {
    auto result = columns_set.insert(column_name);
    if (result.second == false) {
      std::string err_msg = dataset_name + ":" + column_param +
                            ": Invalid parameter, duplicate column names are not allowed: " + *result.first;
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards, int32_t shard_id) {
  if (shuffle) {
    if (num_shards > 1) {
      // If shuffle enabled, sharding enabled, use distributed random sampler
      return DistributedSampler(num_shards, shard_id, shuffle, num_samples);
    }
    // If shuffle enabled, sharding disabled, use random sampler
    return RandomSampler(num_samples >= 0, num_samples);
  }
  if (num_shards > 1) {
    // If shuffle disabled, sharding enabled, use distributed sequential sampler
    return DistributedSampler(num_shards, shard_id, shuffle, num_samples);
  }
  // If shuffle disabled, sharding disabled, use sequential sampler
  return SequentialSampler(0, num_samples);
}

Status DatasetNode::AddCacheOp(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  if (cache_ != nullptr) {
    RETURN_IF_NOT_OK(cache_->Build());
    std::shared_ptr<DatasetOp> cache_op;
    RETURN_IF_NOT_OK(cache_->CreateCacheOp(num_workers_, &cache_op));
    node_ops->push_back(cache_op);
  }
  return Status::OK();
}
// Constructor to initialize the cache
DatasetNode::DatasetNode(const std::shared_ptr<DatasetCache> &dataset_cache) : DatasetNode() { cache_ = dataset_cache; }

std::shared_ptr<DatasetNode> DatasetNode::SetNumWorkers(int32_t num_workers) {
  num_workers_ = num_workers;
  return shared_from_this();
}

DatasetNode::DatasetNode() : cache_(nullptr), parent_(nullptr), children_({}), dataset_size_(-1) {
  // Fetch some default value from config manager
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  connector_que_size_ = cfg->op_connector_size();
  worker_connector_size_ = cfg->worker_connector_size();
  mappable_ = kNotADataSource;
}

std::string DatasetNode::PrintColumns(const std::vector<std::string> &columns) const {
  std::string me;
  if (columns.empty()) {
    me = "<nil>";
  } else {
    me = "[";
    auto i = 0;
    for (auto it = columns.begin(); it < columns.end(); ++it, ++i) {
      me += *it;
      if (i < columns.size() - 1) {
        me += ", ";
      } else {
        me += "]";
      }
    }
  }
  return me;
}

void DatasetNode::PrintTree(std::ostream &out) const {
  int level = 0;
  PrintNode(out, &level);
}

void DatasetNode::PrintNode(std::ostream &out, int *level) const {
  const std::string prefix = "+-";
  const std::string indent = "  ";
  out << prefix;
  Print(out);
  for (const auto &c : this->Children()) {
    out << '\n';
    ++(*level);
    for (auto i = 0; i < *level; i++) {
      out << indent;
    }
    c->PrintNode(out, level);
    --(*level);
  }
}

// Add a node as a child, node's parent needs to be empty
// This function will allow child to be a nullptr, in which case it will simply skip.
// This function is used only when building IR node one by one from parsing the user code.
// During the parsing, we allow a node to have more than one parent, possibly forming a graph.
// It does not maintain the parent_ attribute of the node, which enforces a single parent and a tree structure.
void DatasetNode::AddChild(std::shared_ptr<DatasetNode> child) {
  if (child != nullptr) {
    children_.push_back(child);
  }
}

// Add the input node to be the next child of this node
// This function is used in doing a deep copy of the IR tree built by parsing the user code.
// This function assumes we walk the tree in DFS left-to-right.
// This is a temporary function to be replaced later by a set of better tree operations.
void DatasetNode::AppendChild(std::shared_ptr<DatasetNode> child) {
  if (child != nullptr) {
    if (child->parent_ != nullptr) {
      MS_LOG(WARNING) << "Adding " + child->Name() + " to " + Name() + " but it already has a parent";
    }
    children_.push_back(child);
    child->parent_ = this;
  }
}

// Add a node as a parent, node's parent needs to be empty (future use)
Status DatasetNode::InsertAbove(std::shared_ptr<DatasetNode> node) {
  CHECK_FAIL_RETURN_UNEXPECTED(node != nullptr, "Inserted node must not be a null pointer.");

  if (node->parent_ != nullptr) {
    DatasetNode *parent = node->parent_;
    for (auto i = parent->children_.size() - 1; i >= 0; --i) {
      if (parent->children_[i] == node) {
        parent->children_[i] = static_cast<std::shared_ptr<DatasetNode>>(this);
      }
    }
  }
  children_.push_back(node);
  node->parent_ = this;

  return Status::OK();
}

// Insert a node as a child of this node
// This node's children become the children of the inserted node.
Status DatasetNode::InsertBelow(std::shared_ptr<DatasetNode> node) {
  CHECK_FAIL_RETURN_UNEXPECTED(node != nullptr, "Inserted node must not be a null pointer.");
  CHECK_FAIL_RETURN_UNEXPECTED(node->children_.empty(), "Inserted node must not have any children.");
  CHECK_FAIL_RETURN_UNEXPECTED(node->parent_ == nullptr, "Inserted node must not have a parent.");

  for (auto child : children_) {
    node->children_.push_back(child);
    child->parent_ = node.get();
  }
  // Then establish the new parent-child relationship with the new parent.
  children_.clear();
  children_.push_back(node);
  node->parent_ = this;
  return Status::OK();
}

// Insert a node as a child next to this node (future use)
Status DatasetNode::InsertAfter(std::shared_ptr<DatasetNode> node) {
  CHECK_FAIL_RETURN_UNEXPECTED(parent_ != nullptr, "This node must have a parent.");
  CHECK_FAIL_RETURN_UNEXPECTED(node->parent_ == nullptr, "Inserted node must not have a parent.");
  auto size = parent_->children_.size();
  // Duplicate the last child to increase the size by 1
  parent_->children_.push_back(parent_->children_[size - 1]);
  // Shift each child to its right until we found the insertion point, then insert the input node
  bool found = false;
  for (auto i = parent_->children_.size() - 2; i >= 0; --i) {
    if (parent_->children_[i].get() != this) {
      parent_->children_[i + 1] = parent_->children_[i];
    } else {
      parent_->children_[i + 1] = node;
      node->parent_ = parent_;
      found = true;
      break;
    }
  }
  CHECK_FAIL_RETURN_UNEXPECTED(!found, "Insertion point not found.");
  return Status::OK();
}

// Remove this node from its parent. Add the child of this node to its parent.
// for now, this remove is limited to node with a single child or no child
Status DatasetNode::Remove() {
  CHECK_FAIL_RETURN_UNEXPECTED(parent_ != nullptr, "Cannot remove root or a node without parent.");
  CHECK_FAIL_RETURN_UNEXPECTED(children_.size() < 2, "Cannot remove node with more than 1 child.");
  if (children_.empty()) {  // I am a leaf node, remove me from my parent's children list
    parent_->children_.erase(std::remove(parent_->children_.begin(), parent_->children_.end(), shared_from_this()),
                             parent_->children_.end());  // removal using "erase remove idiom"
  } else {  // replace my position in my parent's children list with my single child
    auto itr = std::find(parent_->children_.begin(), parent_->children_.end(), shared_from_this());
    CHECK_FAIL_RETURN_UNEXPECTED(itr != parent_->children_.end(), "I am not in my parent's children list.");
    children_[0]->parent_ = parent_;  // set my single child's parent ptr to my parent
    *itr = std::move(children_[0]);   // replace me in my parent's children list with my single child
    children_.clear();                // release my single child from my children list
  }
  parent_ = nullptr;
  return Status::OK();
}

// In DFS tree traversal, each node is visited twice. Accept is called on the first visit.
Status DatasetNode::Accept(IRNodePass *p, bool *modified) {
  // This method will only be called if its derived class does not implement one.
  return p->Visit(shared_from_this(), modified);
}

// In DFS tree traversal, each node is visited twice. AcceptAfter is called on the second visit
// after all child nodes are visited.
Status DatasetNode::AcceptAfter(IRNodePass *p, bool *modified) {
  // This method will only be called if its derived class does not implement one.
  return p->VisitAfter(shared_from_this(), modified);
}

Status DatasetNode::GetShardId(int32_t *shard_id) {
  if (!Children().empty()) {
    // Get shard id from the child node
    return Children()[0]->GetShardId(shard_id);
  } else {
    RETURN_STATUS_SYNTAX_ERROR("Get Shard Id failed at source node: " + Name() + "\n");
  }
}

// Gets the dataset size
Status DatasetNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                   int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  if (!IsSizeDefined()) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), dataset_size));
    dataset_size_ = *dataset_size;
    return Status::OK();
  }
  if (children_.size() == 1) {
    return children_[0]->GetDatasetSize(size_getter, estimate, dataset_size);
  } else if (children_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetDatasetSize shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in front of the child_ structure, so we get the dataset size from the last child.
    return children_[children_.size() - 1]->GetDatasetSize(size_getter, estimate, dataset_size);
  } else {
    RETURN_STATUS_UNEXPECTED("Trying to get dataset size from leaf node, missing override");
  }
}
Status DatasetNode::ValidateParams() {
  int32_t num_threads = GlobalContext::config_manager()->num_cpu_threads();
  // in case std::thread::hardware_concurrency returns 0, use an artificial upper limit
  num_threads = num_threads > 0 ? num_threads : std::numeric_limits<uint16_t>::max();
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_workers_ > 0 && num_workers_ <= num_threads,
    Name() + "'s num_workers=" + std::to_string(num_workers_) +
      ", this value is not within the required range of [1, cpu_thread_cnt=" + std::to_string(num_threads) + "].");
  return Status::OK();
}

Status MappableSourceNode::Accept(IRNodePass *p, bool *modified) {
  return p->Visit(shared_from_base<MappableSourceNode>(), modified);
}

Status NonMappableSourceNode::Accept(IRNodePass *p, bool *modified) {
  return p->Visit(shared_from_base<MappableSourceNode>(), modified);
}

}  // namespace dataset
}  // namespace mindspore
