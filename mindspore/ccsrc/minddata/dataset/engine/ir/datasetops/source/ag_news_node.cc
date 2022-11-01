/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/ag_news_node.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/ag_news_op.h"
#include "minddata/dataset/engine/datasetops/source/csv_op.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {
// Constructor for AGNewsNode.
AGNewsNode::AGNewsNode(const std::string &dataset_dir, int64_t num_samples, ShuffleMode shuffle,
                       const std::string &usage, int32_t num_shards, int32_t shard_id,
                       const std::shared_ptr<DatasetCache> &cache)
    : NonMappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id),
      usage_(usage),
      ag_news_files_list_(WalkAllFiles(usage, dataset_dir)) {
  GlobalContext::config_manager()->set_num_shards_for_auto_num_workers(num_shards_);
}

std::shared_ptr<DatasetNode> AGNewsNode::Copy() {
  auto node =
    std::make_shared<AGNewsNode>(dataset_dir_, num_samples_, shuffle_, usage_, num_shards_, shard_id_, cache_);
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void AGNewsNode::Print(std::ostream &out) const {
  out << (Name() + "(cache: " + ((cache_ != nullptr) ? "true" : "false") +
          ", num_shards: " + std::to_string(num_shards_) + ", shard_id: " + std::to_string(shard_id_) + ")");
}

Status AGNewsNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("AGNewsDataset", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("AGNewsDataset", usage_, {"train", "test", "all"}));
  RETURN_IF_NOT_OK(ValidateScalar("AGNewsDataset", "num_samples", num_samples_, {0}, false));
  RETURN_IF_NOT_OK(ValidateDatasetShardParams("AGNewsDataset", num_shards_, shard_id_));
  RETURN_IF_NOT_OK(ValidateEnum("AGNewsDataset", "ShuffleMode", shuffle_,
                                {ShuffleMode::kFalse, ShuffleMode::kFiles, ShuffleMode::kGlobal}));

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("AGNewsDataset", "column_names", column_names_));
  }
  return Status::OK();
}

// Function to build AGNewsNode.
Status AGNewsNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  // Sort the dataset files in a lexicographical order.
  std::vector<std::string> sorted_dataset_files = ag_news_files_list_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());
  // Because AGNews does not have external column_defaults nor column_names parameters,
  // they need to be set before AGNewsOp is initialized.
  // AGNews data set is formatted as three columns of data, so three columns are added.
  std::vector<std::shared_ptr<AGNewsOp::BaseRecord>> column_default;
  column_default.push_back(std::make_shared<CsvOp::Record<std::string>>(AGNewsOp::STRING, ""));
  column_default.push_back(std::make_shared<CsvOp::Record<std::string>>(AGNewsOp::STRING, ""));
  column_default.push_back(std::make_shared<CsvOp::Record<std::string>>(AGNewsOp::STRING, ""));
  std::vector<std::string> column_name = {"index", "title", "description"};
  // AGNews data values are always delimited by a comma.
  char field_delim_ = ',';
  std::shared_ptr<AGNewsOp> ag_news_op =
    std::make_shared<AGNewsOp>(num_workers_, num_samples_, worker_connector_size_, connector_que_size_, shuffle_files,
                               num_shards_, shard_id_, field_delim_, column_default, column_name, sorted_dataset_files);
  RETURN_IF_NOT_OK(ag_news_op->Init());
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp.
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;
    // First, get the number of rows in the dataset.
    RETURN_IF_NOT_OK(AGNewsOp::CountAllFileRows(ag_news_files_list_, false, &num_rows));
    // Add the shuffle op after this op.
    RETURN_IF_NOT_OK(
      AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_, &shuffle_op));
    shuffle_op->SetTotalRepeats(GetTotalRepeats());
    shuffle_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
    node_ops->push_back(shuffle_op);
  }
  ag_news_op->SetTotalRepeats(GetTotalRepeats());
  ag_news_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(ag_news_op);
  return Status::OK();
}

// Get the shard id of node.
Status AGNewsNode::GetShardId(int32_t *shard_id) {
  *shard_id = shard_id_;
  return Status::OK();
}

// Get Dataset size.
Status AGNewsNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(AGNewsOp::CountAllFileRows(ag_news_files_list_, false, &num_rows));
  sample_size = num_samples_;
  num_rows = static_cast<int64_t>(ceil(num_rows / (1.0 * num_shards_)));
  *dataset_size = sample_size > 0 ? std::min(num_rows, sample_size) : num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status AGNewsNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
  args["num_samples"] = num_samples_;
  args["shuffle"] = shuffle_;
  args["num_shards"] = num_shards_;
  args["shard_id"] = shard_id_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

// Note: The following two functions are common among NonMappableSourceNode and
// should be promoted to its parent class. AGNews (for which internally is based off CSV)
// by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree,
// that cache can inherit this sampler from the leaf, providing sampling support from
// the caching layer.
// Should be promoted to its parent class.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status AGNewsNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  *sampler = SelectSampler(num_samples_, shuffle_files, num_shards_, shard_id_);
  return Status::OK();
}

// If a cache has been added into the ascendant tree over this AGNews node, then
// the cache will be executing a sampler for fetching the data.  As such, any
// options in the AGNews node need to be reset to its defaults so that this
// AGNews node will produce the full set of data into the cache.
Status AGNewsNode::MakeSimpleProducer() {
  shard_id_ = 0;
  num_shards_ = 1;
  shuffle_ = ShuffleMode::kFalse;
  num_samples_ = 0;
  return Status::OK();
}

std::vector<std::string> AGNewsNode::WalkAllFiles(const std::string &usage, const std::string &dataset_dir) {
  std::vector<std::string> ag_news_files_list;
  Path train_prefix("train.csv");
  Path test_prefix("test.csv");
  Path dir(dataset_dir);

  if (usage == "train") {
    Path temp_path = dir / train_prefix;
    ag_news_files_list.push_back(temp_path.ToString());
  } else if (usage == "test") {
    Path temp_path = dir / test_prefix;
    ag_news_files_list.push_back(temp_path.ToString());
  } else {
    Path temp_path = dir / train_prefix;
    ag_news_files_list.push_back(temp_path.ToString());
    Path temp_path1 = dir / test_prefix;
    ag_news_files_list.push_back(temp_path1.ToString());
  }
  return ag_news_files_list;
}
}  // namespace dataset
}  // namespace mindspore
