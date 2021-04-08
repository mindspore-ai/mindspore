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
  RETURN_IF_NOT_OK(sampler->ValidateParams());
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
      return DistributedSampler(num_shards, shard_id, shuffle, num_samples).Parse();
    }
    // If shuffle enabled, sharding disabled, use random sampler
    return RandomSampler(num_samples >= 0, num_samples).Parse();
  }
  if (num_shards > 1) {
    // If shuffle disabled, sharding enabled, use distributed sequential sampler
    return DistributedSampler(num_shards, shard_id, shuffle, num_samples).Parse();
  }
  // If shuffle disabled, sharding disabled, use sequential sampler
  return SequentialSampler(0, num_samples).Parse();
}

// Constructor to initialize the cache
DatasetNode::DatasetNode(const std::shared_ptr<DatasetCache> &dataset_cache) : DatasetNode() { cache_ = dataset_cache; }

std::shared_ptr<DatasetNode> DatasetNode::SetNumWorkers(int32_t num_workers) {
  num_workers_ = num_workers;
  return shared_from_this();
}

std::shared_ptr<DatasetNode> DatasetNode::SetDatasetCache(const std::shared_ptr<DatasetCache> &cache) {
  cache_ = cache;
  return shared_from_this();
}

DatasetNode::DatasetNode()
    : cache_(nullptr),
      parent_(nullptr),
      children_({}),
      dataset_size_(-1),
      mappable_(kNotADataSource),
      nary_op_(false),
      descendant_of_cache_(false),
      total_repeats_(-1),
      num_epochs_(1) {
  // Fetch some default value from config manager
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  connector_que_size_ = cfg->op_connector_size();
  worker_connector_size_ = cfg->worker_connector_size();
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
  const std::string indent = "| ";
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

/*
 * AppendChild(<node>) appending <node> as the last child of this node. The new node must have no parent.
 *
 * Input tree:
 *      ds4
 *     /   \
 *   ds3   ds2
 *     |
 *    ds1
 *
 * ds4->AppendChild(ds6) yields this tree
 *
 *      _ ds4 _
 *     /   |   \
 *   ds3  ds2  ds6
 *    |
 *   ds1
 *
 */
Status DatasetNode::AppendChild(std::shared_ptr<DatasetNode> child) {
  CHECK_FAIL_RETURN_UNEXPECTED(IsOrphanNode(child), "Node to append must be an orphan node.");
  CHECK_FAIL_RETURN_UNEXPECTED((IsUnaryOperator() && Children().empty()) || IsNaryOperator(),
                               "This node must be a unary operator with no child or an n-ary operator");
  children_.push_back(child);
  child->parent_ = this;
  return Status::OK();
}

/*
 * InsertChildAt(<pos>, <node>) inserts the <node> to be at the <pos> index of the vector of its child nodes.
 * As in the convention of C++, <pos> starts at position 0.
 * If the <pos> is a negative number or larger than the size of the vector minus one, an error is raised.
 */
Status DatasetNode::InsertChildAt(int32_t pos, std::shared_ptr<DatasetNode> child) {
  CHECK_FAIL_RETURN_UNEXPECTED(pos > -1 && pos <= children_.size(), "Position must in the range of [0, size]");
  CHECK_FAIL_RETURN_UNEXPECTED(IsOrphanNode(child), "Node to append must be an orphan node.");
  CHECK_FAIL_RETURN_UNEXPECTED((IsUnaryOperator() && Children().empty()) || IsNaryOperator(),
                               "This node must be a unary operator with no child or an n-ary operator");
  children_.insert(children_.begin() + pos, child);
  child->parent_ = this;
  return Status::OK();
}

/*
 * Insert the input <node> above this node
 * Input tree:
 *       ds4
 *      /   \
 *     ds3  ds2
 *      |
 *     ds1
 *
 * Case 1: If we want to insert a new node ds5 between ds4 and ds3, use
 *           ds3->InsertAbove(ds5)
 *
 *       ds4
 *      /   \
 *     ds5  ds2
 *      |
 *     ds3
 *      |
 *     ds1
 *
 * Case 2: Likewise, ds2->InsertAbove(ds6) yields
 *
 *       ds4
 *      /   \
 *     ds3  ds6
 *      |    |
 *     ds1  ds2
 *
 * Case 3: We can insert a new node between ds3 and ds1 by ds1->InsertAbove(ds7)
 *
 *       ds4
 *      /   \
 *     ds3  ds2
 *      |
 *     ds7
 *      |
 *     ds1
 *
 * InsertAbove() cannot use on the root node of a tree.
 */
Status DatasetNode::InsertAbove(std::shared_ptr<DatasetNode> node) {
  CHECK_FAIL_RETURN_UNEXPECTED(IsOrphanNode(node), "Node to insert must be an orphan node.");
  CHECK_FAIL_RETURN_UNEXPECTED(parent_ != nullptr, "This node must not be the root or a node without parent.");
  auto parent = parent_;

  // The following fields of these three nodes are changed in this function:
  // 1. parent->children_
  // 2. node->parent_ and node->children_
  // 3. this->parent_
  auto current_node_itr = std::find(parent_->children_.begin(), parent_->children_.end(), shared_from_this());
  *current_node_itr = node;  // replace me in my parent's children list with the newly inserted node
  node->parent_ = parent;    // set the newly inserted node's parent ptr to my parent
  node->children_.push_back(shared_from_this());  // add myself to the newly inserted node's children list
  parent_ = node.get();                           // set my parent ptr to the newly inserted node

  return Status::OK();
}

/*
 * Drop() detaches this node from the tree it is in. Calling Drop() from a standalone node is a no-op.
 *
 * Input tree:
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds8 ds5 ds4 ds1
 *     |     /  \
 *    ds7  ds3  ds2
 *
 * Case 1: When the node has no child and no sibling, Drop() detaches the node from its tree.
 *
 *   ds7->Drop() yields the tree below:
 *
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds8 ds5 ds4 ds1
 *           /  \
 *         ds3  ds2
 *
 * Case 2: When the node has one child and no sibling, Drop() detaches the node from its tree and the node's child
 *         becomes its parent's child.
 *
 *   ds8->Drop() yields the tree below:
 *
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds7 ds5 ds4 ds1
 *           /  \
 *         ds3  ds2
 *
 * Case 3: When the node has more than one child and no sibling, Drop() detaches the node from its tree and the node's
 *         children become its parent's children.
 *
 *   When the input tree is
 *
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |      |
 *    ds8    ds4
 *     |    /   \
 *    ds7  ds3  ds2
 *
 *    ds4->Drop() yields the tree below:
 *
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |     /  \
 *    ds8  ds3  ds2
 *     |
 *    ds7
 *
 *   But if ds6 is not an n-ary operator, ds4->Drop() will raise an error because we cannot add the children of an
 *   n-ary operator (ds4) to a unary operator (ds6).
 *
 * Case 4: When the node has no child but has siblings, Drop() detaches the node from its tree and its siblings will be
 *         squeezed left.
 *
 * Input tree:
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds8 ds5 ds4 ds1
 *     |     /  \
 *    ds7  ds3  ds2
 *
 *   ds5->Drop() yields the tree below:
 *
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |     /  \
 *    ds8   ds4 ds1
 *     |    /  \
 *    ds7 ds3  ds2
 *
 * Case 5: When the node has only one child but has siblings, Drop() detaches the node from its tree and the node's
 *         children become its parent's children.
 *
 * Input tree:
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds8 ds5 ds4 ds1
 *     |      |
 *    ds7     ds3
 *
 *   ds4->Drop() yields the tree below:
 *
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds8 ds5 ds3 ds1
 *     |
 *    ds7
 *
 * Case 6: When the node has more than one child and more than one sibling, Drop() will raise an error.
 *         If we want to drop ds4 from the input tree, ds4->Drop() will not work. We will have to do it
 *         with a combination of Drop(), InsertChildAt()
 *
 * Input tree:
 *       ds10
 *      /    \
 *    ds9    ds6
 *     |   /  |  \
 *    ds8 ds5 ds4 ds1
 *     |     /  \
 *    ds7  ds3  ds2
 *
 * If we want to form this tree below:
 *
 *       ds10
 *      /    \
 *    ds9    ds6_____
 *     |   /  |   |  \
 *    ds8 ds5 ds3 ds2 ds1
 *     |
 *    ds7
 *
 */
Status DatasetNode::Drop() {
  CHECK_FAIL_RETURN_UNEXPECTED(parent_ != nullptr, "This node to drop must not be the root or a node without parent.");
  CHECK_FAIL_RETURN_UNEXPECTED(!(IsNaryOperator() && parent_->IsUnaryOperator()),
                               "Trying to drop an n-ary operator that is a child of a unary operator");
  CHECK_FAIL_RETURN_UNEXPECTED(!(children_.size() > 1 && parent_->children_.size() > 1),
                               "This node to drop must not have more than one child and more than one sibling.");
  if (parent_->children_.size() == 1) {
    auto parent = parent_;
    // Case 2: When the node has one child and no sibling, Drop() detaches the node from its tree and the node's child
    //         becomes its parent's child.
    // This is the most common use case.
    if (children_.size() == 1) {
      auto child = children_[0];
      // Move its child to be its parent's child
      parent->children_[0] = child;
      child->parent_ = parent;
    } else if (children_.empty()) {
      // Case 1: When the node has no child and no sibling, Drop() detaches the node from its tree.
      // Remove this node from its parent's child
      parent_->children_.clear();
    } else if (children_.size() > 1) {
      // Case 3: When the node has more than one child and no sibling, Drop() detaches the node from its tree and
      //         the node's children become its parent's children.
      // Remove this node from its parent's child
      parent->children_.clear();
      // Move its child to be its parent's child
      for (auto &child : children_) {
        parent->children_.push_back(child);
        child->parent_ = parent;
      }
    }
    // And mark itself as an orphan
    parent_ = nullptr;
    children_.clear();
  } else if (children_.empty() && parent_->children_.size() > 1) {
    // Case 4: When the node has no child but has siblings, Drop() detaches the node from its tree and its siblings will
    //         be squeezed left.
    auto parent = parent_;
    // Remove this node from its parent's child
    parent->children_.erase(std::remove(parent->children_.begin(), parent->children_.end(), shared_from_this()),
                            parent->children_.end());  // removal using "erase remove idiom"
    // And mark itself as an orphan
    parent_ = nullptr;
    children_.clear();
  } else if (children_.size() == 1 && parent_->children_.size() > 1) {
    // Case 5: When the node has only one child but has siblings, Drop() detaches the node from its tree and the node's
    //         children become its parent's children.
    auto itr = std::find(parent_->children_.begin(), parent_->children_.end(), shared_from_this());
    CHECK_FAIL_RETURN_UNEXPECTED(itr != parent_->children_.end(), "I am not in my parent's children list.");
    *itr = children_[0];              // replace this node in its parent's children list with its single child
    children_[0]->parent_ = parent_;  // set its single child's parent ptr to its parent
    // And mark itself as an orphan
    parent_ = nullptr;
    children_.clear();
  } else {
    RETURN_STATUS_UNEXPECTED("Internal error: we should not reach here.");
  }
  return Status::OK();
}

// In DFS tree traversal, each node is visited twice. Accept is called on the first visit.
Status DatasetNode::Accept(IRNodePass *const p, bool *const modified) {
  // This method will only be called if its derived class does not implement one.
  return p->Visit(shared_from_this(), modified);
}

// In DFS tree traversal, each node is visited twice. AcceptAfter is called on the second visit
// after all child nodes are visited.
Status DatasetNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // This method will only be called if its derived class does not implement one.
  return p->VisitAfter(shared_from_this(), modified);
}

Status DatasetNode::GetShardId(int32_t *const shard_id) {
  if (children_.size() == 1) {
    // Get shard id from the child node
    return children_[0]->GetShardId(shard_id);
  } else if (children_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetShardId shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in front of the child_ structure, so we get the dataset size from the last child.
    return children_.back()->GetShardId(shard_id);
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
    return children_.front()->GetDatasetSize(size_getter, estimate, dataset_size);
  } else if (children_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetDatasetSize shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in front of the child_ structure, so we get the dataset size from the last child.
    return children_.back()->GetDatasetSize(size_getter, estimate, dataset_size);
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
      "(default=8), this value is not within the required range of [1, cpu_thread_cnt=" + std::to_string(num_threads) +
      "], you can modify num_workers in script like:\n" + Name() + "(data_path, num_parallel_workers=4)");
  return Status::OK();
}

Status DatasetNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  *out_json = args;
  return Status::OK();
}

Status MappableSourceNode::Accept(IRNodePass *const p, bool *const modified) {
  return p->Visit(shared_from_base<MappableSourceNode>(), modified);
}

Status NonMappableSourceNode::Accept(IRNodePass *const p, bool *const modified) {
  return p->Visit(shared_from_base<NonMappableSourceNode>(), modified);
}

}  // namespace dataset
}  // namespace mindspore
